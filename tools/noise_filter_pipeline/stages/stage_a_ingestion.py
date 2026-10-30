"""
Stage A: Source Ingestion

从 dataset registry 读取数据集定义，将原始音频统一转换为 AudioRecord，
输出 A_source_manifest.jsonl。

当前支持类型：
  - directory: 递归扫描音频文件，从文件名/目录/映射表提取标签
  - json: 读取 JSON 文件列表（待实现）
  - csv: 读取 CSV metadata（待实现）
  - huggingface: 读取 HuggingFace Dataset（待实现）
"""
from __future__ import annotations

import os
import io
import json
import csv
import logging
import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Set, Tuple

import soundfile as sf
from tqdm import tqdm

from dataclasses import asdict

from ..core.pipeline_config import UnifiedPipelineConfig
from ..core.registry import AUDIO_EXTS, DatasetRegistry, DatasetEntry
from ..core.manifest_io import AudioRecord, write_manifest, config_hash
from ..core.manifest_per_dataset import reject_legacy_single_manifest
from ..core.stats import write_stage_stats

logger = logging.getLogger(__name__)

# 元信息缓存的中间落盘粒度：生成器尾部一次性写回时，消费方提前中断（异常/
# 取消/后续数据集失败）会丢掉本轮全部 sf.info 结果；按条目数或时长先落盘，
# 把可能的重复扫描量限制在一个间隔内。次数过多会放大全量 JSON 重写 I/O，
# 因此两个阈值都取偏保守的值。
_CACHE_FLUSH_ENTRIES = 20000
_CACHE_FLUSH_INTERVAL_SEC = 120.0

# 进度条预统计时跨数据集缓存的路径数上限。超过后对应数据集不缓存，
# ingest 阶段重新 glob（多一次目录扫描，换取内存有界）；单个数据集
# ingest 时仍需完整 files 列表用于排序与 ratio shuffle，这是 split
# 语义决定的单数据集峰值，不在此上限内。
_PREFETCH_MAX_TOTAL_FILES = 200_000

# sf.info 并行读取的分块大小。Executor.map 会一次性提交全部任务，
# 分块后同时在途的 future/结果数有界，避免百万级文件时元信息驻留 GB 级。
_INFO_CHUNK_SIZE = 8192

# HuggingFace Audio 列按批读取 Arrow 原始 path/bytes 的行数上限。绕开
# datasets/torchcodec 解码后，批大小约束单次转 Python dict 的音频字节量。
_HF_RAW_BATCH_SIZE = 256


def _audio_info_cache_path(output_dir: str, entry_id: str) -> str:
    """返回数据集音频元信息缓存路径。"""
    cache_dir = os.path.join(output_dir, ".stage_a_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{entry_id}_audio_info.json")


def _load_audio_info_cache(cache_path: str) -> Dict[str, Any]:
    """加载缓存；缓存损坏时返回空字典。"""
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.debug("Stage A 缓存加载失败 %s: %s", cache_path, e)
    return {}


def _save_audio_info_cache(cache_path: str, cache_data: Dict[str, Any]) -> None:
    """保存缓存到磁盘（tmp + os.replace 原子替换）。

    直接 "w" 写入时进程被杀/磁盘满会留下半截 JSON，下次加载整份缓存报废
    （_load_audio_info_cache 解析失败返回空，百万级文件全部重扫）；
    tmp+replace 保证磁盘上要么是新缓存完整版、要么是旧缓存。
    """
    tmp_path = cache_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            # 百万级条目下 indent 会显著放大文件体积与读写耗时，使用紧凑格式
            json.dump(cache_data, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    except Exception as e:
        logger.warning("Stage A 缓存保存失败 %s: %s", cache_path, e)
        # 失败时清理半截 tmp，避免残留文件误被当作有效缓存
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _prune_audio_info_cache(cache_data: Dict[str, Any], keep_keys: Set[str]) -> int:
    """剔除指向已消失文件的缓存条目，返回删除条数。

    数据集删文件/改路径后旧条目永不命中，百万级数据集下会让缓存 JSON
    无界增长；本轮刚刷新过的条目（keep_keys）不再重复 stat。
    """
    removed = 0
    for key, entry in list(cache_data.items()):
        if key in keep_keys:
            continue
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path or not os.path.isfile(str(path)):
            del cache_data[key]
            removed += 1
    return removed


def _file_fingerprint(fpath: Path) -> str:
    """基于文件路径、mtime、size 生成缓存 key。"""
    try:
        stat = fpath.stat()
        raw = f"{os.path.abspath(fpath)}|{stat.st_mtime}|{stat.st_size}"
    except OSError:
        raw = str(os.path.abspath(fpath))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# 音频元信息 sanity 上界默认值：超过 6 小时的 duration 基本可判定为
# sf.info 对 VBR/损坏文件的误判，直接按读取失败处理，避免污染下游统计。
# 长时田野录音可通过 stage_a.max_plausible_duration_hours 配置放宽。
_MAX_PLAUSIBLE_DURATION_SEC = 6 * 3600.0


def _is_plausible_audio_info(
    sr: Optional[int],
    duration: Optional[float],
    channels: Optional[int],
    max_duration_sec: Optional[float] = None,
) -> bool:
    """校验 sf.info 返回的元信息是否在合理范围内。

    max_duration_sec: duration 上界；None 时使用模块默认值
    ``_MAX_PLAUSIBLE_DURATION_SEC``。
    """
    if max_duration_sec is None:
        max_duration_sec = _MAX_PLAUSIBLE_DURATION_SEC
    if sr is None or duration is None or channels is None:
        return False
    if sr <= 0 or channels <= 0:
        return False
    if not (0.0 < duration <= max_duration_sec):
        return False
    return True


def _get_audio_info(
    fpath: Path,
    cache_data: Dict[str, Any],
    max_duration_sec: Optional[float] = None,
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[str], Optional[str]]:
    """获取音频元信息，优先使用缓存；失败或数值异常返回 None。

    返回 (sr, duration, channels, key, error_reason)：成功时 error_reason
    为 None；失败时区分 "read_failed"（sf.info 读不出）与 "implausible"
    （读出但数值超合理范围），供调用方写入分型负缓存。

    逐文件失败只打 debug 日志，由调用方聚合成一条汇总 warning，
    避免百万级文件场景刷屏。
    """
    key = _file_fingerprint(fpath)
    cached = cache_data.get(key)
    if cached and isinstance(cached, dict):
        # 负缓存：上次已判定失败，分型决定是否信任，避免无谓的重复 sf.info
        if cached.get("error"):
            reason = cached.get("reason")
            # 读取失败类负条目保持信任（文件未变则重读大概率仍失败）
            if reason == "read_failed":
                return None, None, None, key, "read_failed"
            # implausible 类负条目记录判定时的 duration 上限；配置上限变化
            # 后不再信任，重新探测（例如调大上限接纳长时田野录音）
            current_limit_hours = (
                max_duration_sec if max_duration_sec is not None else _MAX_PLAUSIBLE_DURATION_SEC
            ) / 3600.0
            if reason == "implausible" and cached.get("limit_hours") == current_limit_hours:
                return None, None, None, key, "implausible"
            # 旧格式负条目（缺 reason/limit_hours）与上限已变的 implausible
            # 条目按不可复用处理，落到下方重新读取（安全方向：重算而非信任）
        sr, duration, channels = (
            cached.get("sr"),
            cached.get("duration"),
            cached.get("channels"),
        )
        if _is_plausible_audio_info(sr, duration, channels, max_duration_sec):
            return sr, duration, channels, key, None
        # 缓存中的异常条目不生效，走重新读取；重读仍异常则拒绝
    try:
        info = sf.info(fpath)
        if not _is_plausible_audio_info(info.samplerate, info.duration, info.channels, max_duration_sec):
            logger.debug(
                "音频元信息异常，按读取失败处理 %s: sr=%s duration=%s channels=%s",
                fpath, info.samplerate, info.duration, info.channels,
            )
            return None, None, None, key, "implausible"
        return info.samplerate, info.duration, info.channels, key, None
    except Exception as e:
        logger.debug("无法读取 %s: %s", fpath, e)
        return None, None, None, key, "read_failed"


class StageAOrchestrator:
    """Stage A orchestrator。"""

    def __init__(self, config: UnifiedPipelineConfig, dataset_ids: Optional[Set[str]] = None):
        self.config = config
        # 数据集过滤：None 表示处理全部 enabled 数据集；指定时只处理列表内的数据集
        self.dataset_ids = set(dataset_ids) if dataset_ids else None
        # Stage A 扫描音频元信息属于 I/O 密集型任务，线程数可配置
        self.num_workers = int(
            (self.config.stage_a or {}).get("num_workers", 8)
        )
        # ratio split 的确定性 shuffle 种子；未配置时用 0 保证可复现
        self.seed = self.config.seed if self.config.seed is not None else 0
        # 音频 duration 合理性上界（小时），默认 6h；长时田野录音可调大，
        # 避免被误判为 sf.info 读取异常而进负缓存永久跳过
        self.max_plausible_duration_sec = float(
            (self.config.stage_a or {}).get("max_plausible_duration_hours", 6.0)
        ) * 3600.0
        # 标签映射 YAML 按路径缓存，避免逐文件重复解析
        self._label_mapping_cache: Dict[str, Dict[str, Any]] = {}
        # 进度条预统计阶段物化的文件列表，ingest 时复用，避免目录双重扫描；
        # 缓存总量受 _PREFETCH_MAX_TOTAL_FILES 限制
        self._prefetched_files: Dict[str, List[Path]] = {}

    def run(
        self,
        input_manifest: Optional[str],
        output_dir: str,
        output_manifest: str,  # 保留以对齐 StageRunner 统一接口；per-dataset 架构下
                               # 输出为 a_manifest_<dataset_id>.jsonl，不使用该路径
        global_pbar: Optional[tqdm] = None,
    ) -> str:
        self.global_pbar = global_pbar

        # registry_path 配置项默认相对于 use_train 根目录解析
        use_train_root = Path(__file__).resolve().parent.parent.parent.parent
        registry_path = self.config.resolve_registry_path(str(use_train_root))
        if not os.path.isfile(registry_path):
            raise ValueError(f"Stage A registry 文件不存在: {registry_path}")
        registry = DatasetRegistry(registry_path)
        enabled_entries = registry.list_enabled()
        if not enabled_entries:
            raise ValueError("Stage A registry 中没有 enabled 数据集")

        # --dataset 是显式处理范围，拼写错误或禁用项必须立即失败，避免写空输出。
        if self.dataset_ids is not None:
            known = {e.id for e in enabled_entries}
            unknown = self.dataset_ids - known
            if unknown:
                raise ValueError(
                    f"Stage A --dataset 包含不存在或未启用的 dataset: {sorted(unknown)}"
                )
            selected_entries = [e for e in enabled_entries if e.id in self.dataset_ids]
        else:
            selected_entries = enabled_entries

        for entry in selected_entries:
            # 相对路径统一按 registry 文件所在目录解析（registry.resolve_path），
            # 与 validate_dataset_registry 保持同一口径
            if entry.type == "directory" and not Path(registry.resolve_path(entry.path)).is_dir():
                raise ValueError(
                    f"Stage A 数据集 {entry.id} 目录不存在: {registry.resolve_path(entry.path)}"
                )

        provenance = {
            "pipeline_version": self.config.pipeline_version,
            "config_hash": config_hash(asdict(self.config)),
            "registry_path": registry_path,
            "output_dir": output_dir,
        }

        # 预先统计 Stage A 输入文件数并加入全局进度 total
        if self.global_pbar is not None:
            total_a = self._count_registry_files(registry)
            self.global_pbar.total = (self.global_pbar.total or 0) + total_a
            self.global_pbar.refresh()

        # 输出目录残留旧格式单文件（a_manifest.jsonl）时显式拒绝：
        # per-dataset 架构不会消费它，静默重建会让旧记录无声消失。
        reject_legacy_single_manifest(output_dir, stage_label="Stage A 输出目录")

        # per-dataset 输出：每个数据集写入独立文件 a_manifest_<dataset_id>.jsonl，
        # 增量跑只更新本次涉及的数据集文件，其余数据集文件保留
        total = 0
        zero_output_ids: List[str] = []
        for entry in selected_entries:
            per_file = os.path.join(output_dir, f"a_manifest_{entry.id}.jsonl")
            written = write_manifest(
                self._ingest_dataset(entry, provenance, registry),
                per_file,
            )
            total += written
            if written == 0:
                zero_output_ids.append(entry.id)

        write_stage_stats(
            manifest_path=output_dir,
            stage_name="A",
            output_path=os.path.join(output_dir, "a_stats.json"),
        )
        logger.info("Stage A 写入 %d 条记录（per-dataset manifest）", total)

        # enabled 数据集 0 产出通常是路径/字段配置错误的信号（例如 json 音频
        # 字段名写错导致全部条目被跳过），单独打一条显眼 warning 列出数据集 id
        if zero_output_ids:
            logger.warning(
                "!!!!!! Stage A 有 %d 个 enabled 数据集本次 0 产出: %s；"
                "请检查数据集路径、input_pattern 与字段配置 !!!!!!",
                len(zero_output_ids), ", ".join(zero_output_ids),
            )

        # 清理残留：删除"不在 enabled 数据集集合"中的 per-dataset 文件。
        # 数据集从 registry 移除/禁用后，其历史 manifest 文件会残留并被下游
        # B/C/D 读到（若未处理完会被意外处理）；enabled 但本次被 --dataset
        # 过滤掉的数据集文件必须保留（历史数据，供下游断点恢复）。
        if os.path.isdir(output_dir):
            enabled_ids = {e.id for e in registry.list_enabled()}
            for fname in os.listdir(output_dir):
                if fname.startswith("a_manifest_") and fname.endswith(".jsonl"):
                    ds = fname[len("a_manifest_"):-len(".jsonl")]
                    if ds not in enabled_ids:
                        try:
                            os.remove(os.path.join(output_dir, fname))
                            logger.info("Stage A 清理已禁用数据集的残留 manifest: %s", fname)
                        except OSError as e:
                            logger.warning("Stage A 清理残留失败 %s: %s", fname, e)

        return output_dir

    def _count_registry_files(self, registry: DatasetRegistry) -> int:
        """统计 Stage A 将要处理的输入文件/样本总数。

        directory 类型会把物化的文件列表缓存到 _prefetched_files，
        供 _ingest_directory 复用，避免目录被 glob 两遍。缓存总量受
        _PREFETCH_MAX_TOTAL_FILES 限制；超出上限的数据集改为 ingest 时
        重新扫描，保证跨数据集的路径驻留内存有界。

        单个数据集 ingest 时仍需完整 files 列表用于排序与 ratio shuffle，
        这是 split 语义决定的峰值；AudioRecord 本身走 generator 流式写出。
        """
        total = 0
        prefetched = 0
        for entry in registry.list_enabled():
            if self.dataset_ids is not None and entry.id not in self.dataset_ids:
                continue
            if entry.type == "directory":
                base_path = Path(registry.resolve_path(entry.path))
                if not base_path.exists():
                    continue
                pattern = entry.input_pattern or "**/*.wav"
                files = [f for f in base_path.glob(pattern) if f.suffix.lower() in AUDIO_EXTS]
                if prefetched + len(files) <= _PREFETCH_MAX_TOTAL_FILES:
                    self._prefetched_files[entry.id] = files
                    prefetched += len(files)
                else:
                    logger.debug(
                        "Stage A 数据集 %s 的 %d 个路径超过预取上限，ingest 时重新扫描",
                        entry.id, len(files),
                    )
                total += len(files)
            elif entry.type == "json":
                json_path = Path(registry.resolve_path(entry.path))
                if not json_path.exists():
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        total += len(data)
                except Exception as exc:
                    logger.warning(
                        "Stage A 统计数据集 %s 的 JSON 失败: %s (%s: %s)",
                        entry.id, json_path, type(exc).__name__, exc,
                    )
            elif entry.type == "csv":
                csv_path = Path(registry.resolve_path(entry.path))
                if not csv_path.exists():
                    continue
                try:
                    with open(csv_path, "r", encoding="utf-8", newline="") as f:
                        total += sum(1 for _ in csv.DictReader(f))
                except Exception as exc:
                    logger.warning(
                        "Stage A 统计数据集 %s 的 CSV 失败: %s (%s: %s)",
                        entry.id, csv_path, type(exc).__name__, exc,
                    )
            elif entry.type == "huggingface":
                # HuggingFace 数据集长度无法快速统计，不加入全局 total；
                # 口径对齐：_ingest_huggingface 侧相应也不 update 全局进度条
                pass
        return total

    def _ingest_dataset(
        self,
        entry: DatasetEntry,
        provenance: Dict[str, Any],
        registry: DatasetRegistry,
    ) -> Iterator[AudioRecord]:
        logger.info("Ingest 数据集: %s (%s)", entry.id, entry.name)

        if entry.type == "directory":
            records = self._ingest_directory(entry, provenance, registry, self.num_workers)
        elif entry.type == "json":
            records = self._ingest_json(entry, provenance, registry)
        elif entry.type == "csv":
            records = self._ingest_csv(entry, provenance, registry)
        elif entry.type == "huggingface":
            records = self._ingest_huggingface(entry, provenance, registry)
        else:
            logger.warning("数据集 %s: 暂不支持的 type=%s", entry.id, entry.type)
            return

        # record_id 去重：全量物化 set（每条几十字节，百万级记录约几十 MB，
        # 换取 O(1) 去重与审计日志，可接受）；命中重复时跳过该条并告警，
        # 不再中止整个数据集——单条脏数据不应拖垮其余记录
        seen_record_ids = set()
        for record in records:
            if record.record_id in seen_record_ids:
                logger.warning(
                    "Stage A 数据集 %s 跳过重复 record_id: %s",
                    entry.id, record.record_id,
                )
                continue
            seen_record_ids.add(record.record_id)
            yield record

    def _ingest_directory(
        self,
        entry: DatasetEntry,
        provenance: Dict[str, Any],
        registry: DatasetRegistry,
        num_workers: int = 8,
    ) -> Iterator[AudioRecord]:
        """扫描目录，生成 AudioRecord。

        使用 ThreadPoolExecutor 并行读取音频元信息，并通过缓存跳过未变更文件，
        加速百万级文件扫描。
        """
        # 相对路径按 registry 文件所在目录解析为绝对路径（与 validator 同口径）
        base_path = Path(registry.resolve_path(entry.path))
        if not base_path.exists():
            logger.error("数据集 %s 路径不存在: %s", entry.id, base_path)
            return

        pattern = entry.input_pattern or "**/*.wav"
        # 优先复用进度条预统计阶段的文件列表，避免重复 glob
        files = self._prefetched_files.pop(entry.id, None)
        if files is None:
            files = [f for f in base_path.glob(pattern) if f.suffix.lower() in AUDIO_EXTS]
        files = sorted(files)
        # ratio split 按索引比例切分；sorted 顺序会让同目录/同类文件聚簇，
        # 导致某些类全进 train、某些类全进 test，必须先确定性 shuffle
        if entry.split_strategy == "ratio":
            random.Random(self.seed).shuffle(files)
        logger.info("数据集 %s: 发现 %d 个音频文件", entry.id, len(files))

        output_dir = provenance.get("output_dir") or "./pipeline_out"
        cache_path = _audio_info_cache_path(output_dir, entry.id)
        cache_data = _load_audio_info_cache(cache_path)
        new_cache_entries: Dict[str, Any] = {}
        last_cache_flush = time.monotonic()

        def flush_cache(force: bool = False) -> None:
            """按条目数/时长把本轮新条目并入缓存并落盘。

            尾部一次性写回时，消费方提前中断会丢掉本轮全部 sf.info 结果；
            分次落盘把重复扫描量限制在一个间隔内（force 供收尾调用）。
            """
            nonlocal last_cache_flush
            if not new_cache_entries:
                return
            if (
                not force
                and len(new_cache_entries) < _CACHE_FLUSH_ENTRIES
                and time.monotonic() - last_cache_flush < _CACHE_FLUSH_INTERVAL_SEC
            ):
                return
            cache_data.update(new_cache_entries)
            new_cache_entries.clear()
            _save_audio_info_cache(cache_path, cache_data)
            last_cache_flush = time.monotonic()

        failed_count = 0
        # ratio split 的 idx 口径：只对成功 ingest 的条目递增（成功序号），
        # 跳过/失败条目不消耗序号，避免比例因跳过而偏离配置；同时跳过条目也
        # 推进全局进度条，保证 total（预统计含全部文件）与 update 次数一致
        success_idx = 0
        with tqdm(
            total=len(files),
            desc=f"Stage A [{entry.id}] 读取元信息",
            leave=False,
        ) as pbar, ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            # 分块提交：executor.map 会一次性提交整块任务，块大小约束了
            # 同时在途的 future 与元信息结果数量
            for start in range(0, len(files), _INFO_CHUNK_SIZE):
                chunk = files[start:start + _INFO_CHUNK_SIZE]
                infos = executor.map(
                    lambda p: _get_audio_info(p, cache_data, self.max_plausible_duration_sec),
                    chunk,
                )
                for fpath, (sr, duration, channels, key, error_reason) in zip(chunk, infos):
                    pbar.update(1)
                    if sr is None or duration is None or channels is None or key is None:
                        # 负缓存：分型记录失败条目（reason + 判定时的 duration 上限），
                        # 下次重跑按类型决定是否信任，不再无脑重复 sf.info
                        if key is not None:
                            new_cache_entries[key] = {
                                "error": True,
                                "reason": error_reason or "read_failed",
                                "limit_hours": self.max_plausible_duration_sec / 3600.0,
                                "path": str(os.path.abspath(fpath)),
                            }
                            flush_cache()
                        failed_count += 1
                        if self.global_pbar is not None:
                            self.global_pbar.update(1)
                        continue
                    # 更新缓存条目
                    new_cache_entries[key] = {
                        "sr": sr,
                        "duration": duration,
                        "channels": channels,
                        "path": str(os.path.abspath(fpath)),
                    }
                    flush_cache()

                    source_id = self._make_source_id(fpath, base_path)
                    labels = self._extract_labels_directory(entry, fpath, base_path, registry)
                    split = self._resolve_split(entry, fpath, success_idx, len(files), registry)

                    record = AudioRecord(
                        record_id=AudioRecord.make_record_id(entry.id, source_id),
                        dataset_id=entry.id,
                        source_id=source_id,
                        chunk_id="source",
                        split=split,
                        audio={
                            "path": str(fpath),
                            "original_path": str(fpath),
                            "sr": sr,
                            "duration": duration,
                            "channels": channels,
                        },
                        labels=labels,
                        provenance=provenance,
                    )
                    record.mark_stage("A", "passed")
                    success_idx += 1
                    yield record
                    if self.global_pbar is not None:
                        self.global_pbar.update(1)

        # 合并旧缓存与新条目，保留未变更文件的旧记录；同时剔除指向已消失
        # 文件的旧条目（数据集删文件/改路径后缓存会无界增长）
        fresh_keys = set(new_cache_entries)
        cache_data.update(new_cache_entries)
        new_cache_entries.clear()
        pruned = _prune_audio_info_cache(cache_data, fresh_keys)
        _save_audio_info_cache(cache_path, cache_data)
        if pruned:
            logger.info("数据集 %s: 清理 %d 条指向已消失文件的缓存条目", entry.id, pruned)

        # 逐文件失败日志已降级为 debug，这里聚合打一条汇总 warning
        if failed_count:
            logger.warning(
                "数据集 %s: %d/%d 个文件读取失败或元信息异常，已按失败处理并写入负缓存"
                "（逐文件明细见 debug 日志）",
                entry.id, failed_count, len(files),
            )

    def _ingest_json(
        self,
        entry: DatasetEntry,
        provenance: Dict[str, Any],
        registry: DatasetRegistry,
    ) -> Iterator[AudioRecord]:
        """读取 JSON 文件列表。

        期望格式：list[dict]，每个 dict 至少含音频路径与标签字段。
        默认字段：audio_path, text_label；可通过 audio_field / label_field 覆盖。
        """
        # 相对路径按 registry 文件所在目录解析为绝对路径（与 validator 同口径）
        json_path = Path(registry.resolve_path(entry.path))
        if not json_path.exists():
            logger.error("数据集 %s JSON 文件不存在: %s", entry.id, json_path)
            return

        audio_field = entry.audio_field or "audio_path"
        label_field = entry.label_field or "text_label"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("数据集 %s JSON 解析失败: %s", entry.id, e)
            return

        if not isinstance(data, list):
            logger.error("数据集 %s JSON 根节点应为列表", entry.id)
            return

        # ratio split 按索引切分，先确定性 shuffle 避免同类聚簇导致 split 偏斜
        if entry.split_strategy == "ratio":
            random.Random(self.seed).shuffle(data)

        total = len(data)
        # ratio split 的 idx 口径：只对成功 ingest 的条目递增（成功序号），
        # 跳过条目不消耗序号；跳过条目同样推进全局进度条，与预统计 total 对齐
        success_idx = 0
        for idx, item in enumerate(data):
            # 条目必须是 dict；混入字符串/数字等坏条目时跳过并告警，
            # 避免 item.get 硬崩导致整个数据集 ingest 中断
            if not isinstance(item, dict):
                logger.warning(
                    "数据集 %s 第 %d 条不是 dict（实际类型 %s），已跳过",
                    entry.id, idx, type(item).__name__,
                )
                if self.global_pbar is not None:
                    self.global_pbar.update(1)
                continue
            raw_audio_path = item.get(audio_field)
            audio_path = self._resolve_metadata_audio_path(raw_audio_path, json_path.parent)
            if audio_path is None or not os.path.isfile(audio_path):
                logger.warning("数据集 %s 第 %d 条音频不存在: %s", entry.id, idx, raw_audio_path)
                if self.global_pbar is not None:
                    self.global_pbar.update(1)
                continue

            labels = item.get(label_field)
            split = self._resolve_split(entry, Path(audio_path), success_idx, total, registry)
            record = self._build_record_from_path(
                entry, audio_path, labels, split, provenance, source_base=json_path.parent
            )
            if record:
                success_idx += 1
                yield record
            if self.global_pbar is not None:
                self.global_pbar.update(1)

    def _ingest_csv(
        self,
        entry: DatasetEntry,
        provenance: Dict[str, Any],
        registry: DatasetRegistry,
    ) -> Iterator[AudioRecord]:
        """读取 CSV metadata。

        默认字段：audio_path, label；可通过 audio_column / label_column 覆盖。
        多标签用 label_separator 分隔。
        """
        # 相对路径按 registry 文件所在目录解析为绝对路径（与 validator 同口径）
        csv_path = Path(registry.resolve_path(entry.path))
        if not csv_path.exists():
            logger.error("数据集 %s CSV 文件不存在: %s", entry.id, entry.path)
            return

        audio_col = entry.audio_column or "audio_path"
        label_col = entry.label_column or "label"

        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            logger.error("数据集 %s CSV 解析失败: %s", entry.id, e)
            return

        # ratio split 按索引切分，先确定性 shuffle 避免同类聚簇导致 split 偏斜
        if entry.split_strategy == "ratio":
            random.Random(self.seed).shuffle(rows)

        total = len(rows)
        # ratio split 的 idx 口径：只对成功 ingest 的条目递增（成功序号），
        # 跳过条目不消耗序号；跳过条目同样推进全局进度条，与预统计 total 对齐
        success_idx = 0
        for idx, row in enumerate(rows):
            raw_audio_path = row.get(audio_col)
            audio_path = self._resolve_metadata_audio_path(raw_audio_path, csv_path.parent)
            if audio_path is None or not os.path.isfile(audio_path):
                logger.warning("数据集 %s 第 %d 行音频不存在: %s", entry.id, idx, raw_audio_path)
                if self.global_pbar is not None:
                    self.global_pbar.update(1)
                continue

            raw_label = row.get(label_col)
            labels = self._parse_label_string(raw_label, entry.label_separator)
            split = self._resolve_split(entry, Path(audio_path), success_idx, total, registry)
            record = self._build_record_from_path(
                entry, audio_path, labels, split, provenance, source_base=csv_path.parent
            )
            if record:
                success_idx += 1
                yield record
            if self.global_pbar is not None:
                self.global_pbar.update(1)

    @staticmethod
    def _iter_hf_samples(
        ds: Any, audio_field: str
    ) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """优先按 Arrow 原始 path/bytes 迭代 Audio 列，绕开 torchcodec 解码。

        datasets 的 Audio feature 在取值时会调用 torchcodec；部分环境（缺少
        libnvrtc/FFmpeg）无法加载。Audio 列在 Arrow 中本就是 struct(path, bytes)，
        直接读取后交给 soundfile 即可。非 Audio 字段或无法访问底层 table 时
        回退到普通迭代。

        ``Dataset.shuffle`` 通过 ``_indices`` 映射实现而非重排 Arrow table，
        因此 raw 迭代必须同时应用该映射，否则 ratio split 的 shuffle 失效。
        """
        features = getattr(ds, "features", None) or {}
        feature = features.get(audio_field)
        table = getattr(getattr(ds, "_data", None), "table", None)
        indices_table = getattr(ds, "_indices", None)
        index_values = None
        if indices_table is not None:
            index_values = indices_table.table.column("indices").to_pylist()
        if getattr(feature, "_type", None) == "Audio" and table is not None:
            total = len(ds)
            for start in range(0, total, _HF_RAW_BATCH_SIZE):
                stop = min(start + _HF_RAW_BATCH_SIZE, total)
                if index_values is not None:
                    rows = table.take(index_values[start:stop]).to_pylist()
                else:
                    rows = table.slice(start, stop - start).to_pylist()
                for offset, row in enumerate(rows):
                    yield start + offset, row
            return
        for idx, sample in enumerate(ds):
            yield idx, sample

    def _ingest_huggingface(
        self,
        entry: DatasetEntry,
        provenance: Dict[str, Any],
        registry: DatasetRegistry,
    ) -> Iterator[AudioRecord]:
        """读取 HuggingFace Dataset。

        将音频保存到 output_dir 下的 hf_cache，再生成 AudioRecord。
        默认字段：audio, labels；可通过 audio_field / label_field 覆盖。
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            logger.error("数据集 %s HuggingFace 加载需要 datasets 库: %s", entry.id, e)
            return

        audio_field = entry.audio_field or "audio"
        label_field = entry.label_field or "labels"

        # 音频缓存目录：放在输出目录同级，避免污染源码树
        cache_dir = Path(provenance.get("output_dir", "./pipeline_out")) / "hf_audio" / entry.id
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            split = entry.splits[0] if entry.splits else "train"
            # 本地路径相对配置时按 registry 目录解析；解析后不是目录则视为
            # HuggingFace 仓库名，load_dataset 必须使用原始字符串（仓库 id
            # 形如 org/name，不能被本地路径解析改写）
            resolved_path = registry.resolve_path(entry.path)
            if os.path.isdir(resolved_path):
                from datasets import DatasetDict, load_from_disk
                ds = load_from_disk(resolved_path)
                if isinstance(ds, DatasetDict):
                    # DatasetDict 且配置的 split 不存在时必须明确报错并列出
                    # 可用 split；否则后续 len/迭代会把 split 名当样本静默消费
                    if split not in ds:
                        logger.error(
                            "数据集 %s HuggingFace split %r 不存在，可用 split: %s",
                            entry.id, split, sorted(ds.keys()),
                        )
                        return
                    ds = ds[split]
            else:
                ds = load_dataset(entry.path, split=split, streaming=False)
        except Exception as e:
            logger.error("数据集 %s HuggingFace 加载失败: %s", entry.id, e)
            return

        # ratio split 按索引切分，先确定性 shuffle 避免同类聚簇导致 split 偏斜
        if entry.split_strategy == "ratio":
            ds = ds.shuffle(seed=self.seed)

        total = len(ds)
        for idx, sample in self._iter_hf_samples(ds, audio_field):
            audio_data = sample.get(audio_field)
            if not audio_data:
                logger.warning("数据集 %s 第 %d 条缺少 audio 字段", entry.id, idx)
                continue

            # audio_data 可能是 Arrow 原始 dict{'path', 'bytes'}、
            # 已解码 dict{'path', 'array', 'sampling_rate'}、tuple
            # 或 datasets >= 2.19 + torchcodec 下的 AudioDecoder 对象。
            if isinstance(audio_data, dict):
                array = audio_data.get("array")
                sr = audio_data.get("sampling_rate", 16000)
                if array is None:
                    raw_bytes = audio_data.get("bytes")
                    raw_path = audio_data.get("path")
                    try:
                        if raw_bytes:
                            array, sr = sf.read(io.BytesIO(raw_bytes))
                        elif raw_path:
                            array, sr = sf.read(raw_path)
                    except Exception as e:
                        logger.warning(
                            "数据集 %s 第 %d 条音频解码失败: %s", entry.id, idx, e
                        )
                        continue
            elif isinstance(audio_data, tuple):
                sr, array = audio_data
            elif hasattr(audio_data, "get_all_samples") and hasattr(audio_data, "metadata"):
                # datasets.Audio 解码器（torchcodec），需手动提取样本
                try:
                    samples = audio_data.get_all_samples()
                    import torch
                    array = samples.data.squeeze().cpu().numpy()
                    sr = int(getattr(samples, "sample_rate", audio_data.metadata.sample_rate))
                except Exception as e:
                    logger.warning("数据集 %s 第 %d 条 AudioDecoder 解码失败: %s", entry.id, idx, e)
                    continue
            else:
                logger.warning(
                    "数据集 %s 第 %d 条 audio 格式不支持: %s",
                    entry.id, idx, type(audio_data).__name__,
                )
                continue

            if array is None:
                continue

            audio_path = cache_dir / f"{idx:08d}.wav"
            try:
                sf.write(str(audio_path), array, sr)
            except Exception as e:
                logger.warning("数据集 %s 第 %d 条音频写入失败: %s", entry.id, idx, e)
                continue

            labels = sample.get(label_field)
            split = self._resolve_split(entry, audio_path, idx, total, registry)
            record = self._build_record_from_path(
                entry, str(audio_path), labels, split, provenance
            )
            # 进度口径说明：HF 数据集未计入全局 total（长度加载前不可知，
            # 见 _count_registry_files），因此这里也不 update 全局进度条，
            # 保持 total/update 口径一致，避免进度条越过 total
            if record:
                yield record

    def _parse_label_string(self, raw_label: Optional[str], separator: Optional[str]) -> Optional[Any]:
        """把 CSV 中的标签字符串解析为字符串或列表。"""
        if raw_label is None:
            return None
        raw_label = raw_label.strip()
        if not raw_label:
            return None
        # label_separator 显式配置为 null/空串时表示不拆分，原样返回整串；
        # 直接 `separator in raw_label` 在 None 时会 TypeError
        if not separator:
            return raw_label
        if separator in raw_label:
            return [s.strip() for s in raw_label.split(separator) if s.strip()]
        return raw_label

    @staticmethod
    def _resolve_metadata_audio_path(raw_path: Any, metadata_dir: Path) -> Optional[str]:
        """将 metadata 中的相对音频路径按 metadata 所在目录解析。"""
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = metadata_dir / path
        return str(path.resolve())

    def _build_record_from_path(
        self,
        entry: DatasetEntry,
        audio_path: str,
        labels: Any,
        split: str,
        provenance: Dict[str, Any],
        source_base: Optional[Path] = None,
    ) -> Optional[AudioRecord]:
        """根据音频路径和标签构造 AudioRecord。"""
        try:
            info = sf.info(audio_path)
            sr = info.samplerate
            duration = info.duration
            channels = info.channels
        except Exception as e:
            logger.warning("无法读取 %s: %s", audio_path, e)
            return None

        fpath = Path(audio_path)
        source_id = self._make_source_id(
            fpath, source_base or (Path(entry.path) if entry.path else fpath.parent)
        )

        record = AudioRecord(
            record_id=AudioRecord.make_record_id(entry.id, source_id),
            dataset_id=entry.id,
            source_id=source_id,
            chunk_id="source",
            split=split,
            audio={
                "path": audio_path,
                "original_path": audio_path,
                "sr": sr,
                "duration": duration,
                "channels": channels,
            },
            labels={"original": labels},
            provenance=provenance,
        )
        record.mark_stage("A", "passed")
        return record

    def _make_source_id(self, fpath: Path, base_path: Path) -> str:
        """生成稳定 source_id：使用相对路径；脱离基准目录时加入路径 hash。"""
        try:
            rel = fpath.resolve().relative_to(base_path.resolve())
            return rel.with_suffix("").as_posix()
        except ValueError:
            digest = hashlib.sha1(str(fpath.resolve()).encode("utf-8")).hexdigest()[:12]
            return f"{fpath.stem}-{digest}"

    def _extract_labels_directory(
        self,
        entry: DatasetEntry,
        fpath: Path,
        base_path: Path,
        registry: DatasetRegistry,
    ) -> Dict[str, Any]:
        """从目录/文件名提取原始标签。"""
        labels: Dict[str, Any] = {"original": None}

        if entry.label_source == "filename":
            labels["original"] = fpath.stem
        elif entry.label_source == "directory":
            try:
                labels["original"] = fpath.relative_to(base_path).parent.name
            except ValueError:
                labels["original"] = "unknown"
        elif entry.label_source == "mapping":
            labels["original"] = self._lookup_label_mapping(entry, fpath, registry)
        elif entry.label_source == "none":
            labels["original"] = None
        else:
            labels["original"] = "unknown"

        return labels

    def _lookup_label_mapping(
        self,
        entry: DatasetEntry,
        fpath: Path,
        registry: DatasetRegistry,
    ) -> Optional[str]:
        """从 YAML 标签映射文件查询标签。

        label_mapping 配置项约定（与 validate_dataset_registry 同口径，
        均以注册表文件所在目录为基准）：
          - 若不含目录分隔符，默认在 <registry 目录>/labels/ 下查找；
          - 若为相对路径，相对于 <registry 目录> 解析。
        """
        if not entry.label_mapping:
            return None

        mapping_path = entry.label_mapping_path(registry._dir)

        if not os.path.isfile(mapping_path):
            logger.warning("标签映射文件不存在: %s", mapping_path)
            return None
        # YAML 映射按路径缓存一次加载；逐文件重复解析在百万级数据下不可接受
        if mapping_path not in self._label_mapping_cache:
            try:
                import yaml
                with open(mapping_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self._label_mapping_cache[mapping_path] = data.get("mappings", {}) or {}
            except Exception as e:
                logger.warning("标签映射读取失败 %s: %s", mapping_path, e)
                self._label_mapping_cache[mapping_path] = {}
        mapping = self._label_mapping_cache[mapping_path]
        label = mapping.get(fpath.stem)
        if label is not None:
            return label
        # ESC-50 文件名形如 ``1-100032-A-0``（fold-clip-take-category），
        # 映射键为类别序号 "1".."50"（= 末段数字 + 1），直接 stem 查不到时按此回退。
        # 限定至少 4 段，且仅当映射键全部为 1..50 的数字类别序号（即确为
        # ESC-50 风格映射）时才启用，避免误伤其它带 label_mapping 的数据集
        # 的数字结尾文件名。
        parts = fpath.stem.split("-")
        is_esc50_style_mapping = bool(mapping) and all(
            k.isdigit() and 1 <= int(k) <= 50 for k in mapping
        )
        if len(parts) >= 4 and is_esc50_style_mapping:
            try:
                class_idx = int(parts[-1]) + 1
            except ValueError:
                return None
            return mapping.get(str(class_idx))
        return None

    def _resolve_split(
        self,
        entry: DatasetEntry,
        fpath: Path,
        idx: int,
        total: int,
        registry: DatasetRegistry,
    ) -> str:
        """根据 split_strategy 确定 split。"""
        if not entry.splits:
            return "train"
        if len(entry.splits) == 1:
            return entry.splits[0]

        strategy = entry.split_strategy
        if strategy == "directory":
            try:
                # entry.path 与 fpath 都按 registry 目录解析后的绝对路径比较
                rel_parts = fpath.relative_to(Path(registry.resolve_path(entry.path))).parts
                if rel_parts[0] in entry.splits:
                    return rel_parts[0]
            except ValueError:
                pass
            return entry.splits[0]

        if strategy == "ratio" and entry.split_ratio:
            ratios = entry.split_ratio
            cum = [sum(ratios[: i + 1]) for i in range(len(ratios))]
            pos = idx / max(1, total)
            for split, threshold in zip(entry.splits, cum):
                if pos < threshold:
                    return split
            return entry.splits[-1]

        return entry.splits[0]
