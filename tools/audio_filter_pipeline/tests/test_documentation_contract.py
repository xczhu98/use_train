"""Audio Filter Pipeline 当前文档契约测试。"""
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).parent.parent
README = PROJECT_ROOT / "README.md"
DATASET_AUDIT = (
    PROJECT_ROOT / "docs" / "dataset_language_sampling_rate_audit_2026-08.md"
)


def test_readme_tracks_current_execution_migration_and_export_contracts():
    text = README.read_text(encoding="utf-8")

    for expected in (
        "regular3",
        "aggregate4",
        "SQLite",
        "有界磁盘排序",
        "`source_decode.v5`",
        "raw cap 512 MiB",
        "总 inflight 上限 4 GiB",
        "每 source 建立并销毁一个 TEN handler",
        "补零到窗口长度",
        'asset_load_binding: {"version": 3}',
        "step3_mos_pool.v6",
        "audio_filter_pipeline.step3_mos_worker.v6",
        "`scoring_execution_policy.v1`",
        "同一 tuple 的队列可跨 source 聚合",
        "32 行、64 MiB float32",
        "close、reverify、再 open",
        "队列层 waveform 不 padding",
        "UTMOS/SCOREQ 的模型内预处理 padding",
        "ordinal_sidecar_path",
        "ordinal_sidecar_sha256",
        "ordinal_sidecar_size_bytes",
        "512 MiB physical JSONL part",
        "自动 rollover",
        "CPU/GPU scorer",
        "part index 链式消费",
        "aggregate 预算下限 4 GiB 并随 row_count 按 4 KiB/行伸缩",
        "owned temporary SQLite",
        "rows 与 bytes 双重约束",
        "两遍扫描",
        "worker input/result/ordinal sidecar",
        "单 worker subshard canonical 上限为 40",
        "request-global ordinal 唯一性",
        "各只打开一次并保留同一 `O_NOFOLLOW` FD",
        "每个 output 创建前",
        "source_package_sha256",
        "verified owned snapshot",
        "default route 不选择 URGENT-PK",
        "`speaker_shadow.v9`",
        "按最多 32 段惰性微分批",
        "descriptor-bound 解码边界",
        "dataset-local scoring binding",
        "仅经 CLI factories 显式传入 `dataset_id` 的生产路径启用",
        "direct 构造不指定 `dataset_id` 时保持全局 digest、可跨 dataset 复用",
        "指定 `dataset_id` 则使用 local binding，并在运行时校验 context",
        "dataset-local Step4/Step6 decision binding",
        "logical `page_count * page_size`、固定 cache、已保留的 key 与当前 key reservation",
        "不使用文件 `getsize`",
        "首通道 mono FLAC/PCM_24",
        "旧资产不自动删除",
        "不泛称所有旧 Step0–Step2 artifact 兼容",
        "exclusion manifest **只支持** `audio_filter_pipeline.enrollment_exclusion.v2`",
        "algorithm` 必须为 `mean_cosine_linear_sum.v1`",
        "algorithm 后缀 `v1` 标识当前 O(nd) 算法版本，不是旧 manifest schema",
        "旧 `enrollment_exclusion.v1` 不兼容",
        "必须从 Step3 重建 enrollment bundle 与 exclusion manifest，并更新配置摘要",
        "不能只改 schema 字段伪装迁移",
        "真实 B×T ONNX batch",
        "GPU SCOREQ 在第 11 次实际 ONNX run 前重建 session",
        "MOS batching verification",
        "CPU 上 3 条真实 SCOREQ 样本的绝对误差为 0",
        "未观察到吞吐改善",
        "UTMOS 的 `utmos_model_sha256` 已同步为当前本地 tree digest `abe1c1aa15225599fa7189a9887da12842f8beb5541add007855ead9acdb6d9a`", "GPU benchmark 尚未运行",
        "tools/verify/benchmark_mos_batching.py",
        "完整匹配当前 request descriptor",
        "不启动 worker",
        "renameat2(RENAME_NOREPLACE)",
        "artifact reader gate 验证已发布 artifact",
        "历史 GPU smoke",
        "不代表本版本的 binding v3、pool v6、worker v6、`scoring_execution_policy.v1` 或 sparse-ordinal 新路径已完成真实 GPU 验证",
        "输出根的优先级固定为 canonical CLI `--output_root`、YAML `output_root`",
    ):
        assert expected in text

    for obsolete in (
        'asset_load_binding: {"version": 2}',
        "step3_mos_pool.v4",
        "audio_filter_pipeline.step3_mos_worker.v4",
        "audio_filter_pipeline.step3_mos_worker.v5",
        "Step5 `speaker_shadow.version` 为 7",
        "Step4/Step6 导出保留全部通道",
        "合法的 Step0–Step2 artifact 可以继续复用",
        "单个 tuple JSONL 与全部 spool 的有界上限均为 4 GiB",
        "GPU LPT metadata 独立限制为 1 GiB",
        "LPT 除 ordinal 数组外",
    ):
        assert obsolete not in text


def test_dataset_audit_tracks_current_versions_route_and_migration_boundary():
    text = DATASET_AUDIT.read_text(encoding="utf-8")

    for expected in (
        "regular3",
        "aggregate4",
        "SQLite",
        "有界磁盘排序",
        "dataset-local scoring binding",
        "dataset-local decision binding",
        "CLI factories 显式传入 `dataset_id` 的生产路径",
        "direct 构造不指定 `dataset_id` 时保持全局 digest、可跨 dataset 复用",
        "指定 `dataset_id` 时使用 local binding，并在运行时校验 context",
        "logical `page_count * page_size`、固定 cache、已保留的 key 与当前 key reservation",
        "不使用文件 `getsize`",
        'asset_load_binding: {"version": 3}',
        "step3_mos_pool.v6",
        "audio_filter_pipeline.step3_mos_worker.v6",
        "ordinal_sidecar_path",
        "ordinal_sidecar_sha256",
        "ordinal_sidecar_size_bytes",
        "512 MiB physical JSONL part",
        "自动 rollover",
        "CPU/GPU scorer",
        "part index 链式消费",
        "aggregate 预算为 4 GiB",
        "owned temporary SQLite",
        "rows 与 bytes 双重约束",
        "两遍扫描",
        "worker input/result/ordinal sidecar",
        "单 worker subshard canonical 上限为 40",
        "request-global ordinal 唯一性",
        "各只打开一次并保留同一 `O_NOFOLLOW` FD",
        "每个 output 创建前",
        "source_package_sha256",
        "verified owned snapshot",
        "default route 不选择 URGENT-PK",
        "Step5 的 `speaker_shadow.version` 为 9",
        "跨最多 32 段",
        "只支持** `audio_filter_pipeline.enrollment_exclusion.v2`",
        "算法必须为 `mean_cosine_linear_sum.v1`",
        "algorithm 后缀 `v1` 表示当前 O(nd) 算法版本，不是旧 manifest schema",
        "旧 `enrollment_exclusion.v1` 不兼容",
        "必须从 Step3 重建 enrollment bundle 与 exclusion manifest，并更新配置摘要",
        "不能只改 schema 字段伪装迁移",
        "历史 GPU smoke",
        "不代表本版本的 binding v3、pool v6、worker v6、持久 request 恢复或 sparse-ordinal 路径已完成真实 GPU 验证",
        "producer identity、config digest、lineage digest 与 row contract 全部精确匹配",
        "旧资产不自动删除",
        "不泛称所有旧 Step0–Step2 artifact 兼容",
        "`audio_filter_{language_bucket}_{rate}.yaml`",
        "`16k`/`32k` 表达 source floor 与固定 output rate",
        "数据集的特殊属性不由 `mixed` route 推断",
        "由各 dataset entry 的 `category` 元数据表达",
    ):
        assert expected in text

    for obsolete in (
        'asset_load_binding: {"version": 2}',
        "step3_mos_pool.v4",
        "audio_filter_pipeline.step3_mos_worker.v4",
        "Step5 的 `speaker_shadow.version` 为 7",
        "合法的 Step0–Step2 artifact 继续复用",
        "默认 SCOREQ + URGENT-PK + DNSMOS 3/3",
        "单个 tuple JSONL 与全部 spool 的有界上限均为 4 GiB",
        "GPU LPT metadata 另设 1 GiB 内存预算",
    ):
        assert obsolete not in text


def test_dataset_audit_tracks_step6_exclusion_inventory():
    text = DATASET_AUDIT.read_text(encoding="utf-8")

    for expected in (
        "36 个 active ID",
        "35 个物理目录",
        "`step6 active IDs = 36`",
        "`step6 physical roots = 35`",
        "排除 clean/filter 已有结果的 6 个",
        "step4 logical IDs = 53",
        "step4 source selectors = 6",
        "`genshinvoice_44k1_spk`",
        "`sample_rates_hz: [48000]`",
    ):
        assert expected in text

    for obsolete in (
        "Step6 speaker inventory 只有 1 个 active ID",
        "`step6 active IDs = 1`",
        "Step6 speaker inventory 仍包含该项",
        "audio_filter_mixed_16k.yaml` 是 `datasets: {}` 空配置模板",
        "Step6 以单一 ID 覆盖两档",
        "`sample_rates_hz: [44100, 48000]`",
    ):
        assert obsolete not in text


def test_pipeline_docs_are_not_ignored_by_git():
    docs = sorted((PROJECT_ROOT / "docs").rglob("*.md"))
    assert docs

    for path in docs:
        result = subprocess.run(
            (
                "git",
                "check-ignore",
                "--no-index",
                "--quiet",
                "--",
                str(path.relative_to(PROJECT_ROOT)),
            ),
            cwd=PROJECT_ROOT,
            check=False,
            timeout=30,
        )
        assert result.returncode == 1, f"文档仍被 Git 忽略：{path}"
