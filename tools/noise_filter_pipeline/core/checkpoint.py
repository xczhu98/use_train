"""
Checkpoint / resume 工具。

为长时运行的 processor（C1/C2/C3/D1/E1 等）提供统一的断点续跑能力：
  - 在输出目录维护 checkpoint.json
  - 记录已处理的 record_id 集合
  - 启动时读取并跳过已处理记录
  - 支持增量写入与原子保存
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from .manifest_io import atomic_replace_durable

logger = logging.getLogger(__name__)


class CheckpointTracker:
    """基于 record_id 集合的断点续跑追踪器。

    支持按 ``save_interval`` 批量落盘：每累计到指定条数时自动保存，
    降低百万级文件场景下频繁写 checkpoint.json 的 I/O 开销。
    最终仍需调用 ``save(force=True)`` 确保尾部数据落盘。

    .. warning::

        ``pre_save_callbacks`` 在 ``_save_lock`` 持有期间同步执行。
        callback 内**禁止**调用 ``save()`` / ``mark_done()`` /
        ``mark_done_batch()``：这些入口都会再次获取同一把非可重入锁
        （或经由 interval 触发 ``save``），同线程会立即自锁死锁。
        callback 只应做无副作用的落盘动作（如 flush manifest）。
    """

    def __init__(
        self,
        checkpoint_path: Optional[str],
        enabled: bool = True,
        save_interval: int = 1,
        pre_save_callbacks: Optional[List[Callable[[], None]]] = None,
        config_hash: Optional[str] = None,
    ):
        self.enabled = enabled and checkpoint_path is not None
        self.checkpoint_path = checkpoint_path
        self.save_interval = max(1, int(save_interval))
        self._pre_save_callbacks = pre_save_callbacks or []
        self.config_hash = config_hash
        self._done: Set[str] = set()
        self._dirty = False
        self._marks_since_save = 0
        self._lock = threading.Lock()
        # save 包含 callback、快照和原子替换，必须串行；否则并发保存会让旧快照
        # 在新快照之后落盘，或同时争用同一个 .tmp 文件。
        self._save_lock = threading.Lock()

        if self.enabled:
            self._load()

    @staticmethod
    def _fingerprint_matches(saved_hash: Optional[str], current_hash: Optional[str]) -> bool:
        """当前配置指纹与 checkpoint 保存的指纹是否严格一致。

        组合指纹必须严格匹配；未启用指纹（current 为 None）时允许恢复。
        """
        if not current_hash:
            return True
        return saved_hash == current_hash

    def _load(self) -> None:
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            return
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            done = data.get("done", [])
            saved_hash = data.get("config_hash")
            # 配置指纹校验：resume 必须基于一致的配置，否则静默恢复旧配置的结果
            # 会让"改配置重跑"失效（旧结果被恢复、新配置不生效）。
            # 不一致/无法验证时视为 checkpoint 无效，拒绝恢复、全量重算。
            if self.config_hash and not self._fingerprint_matches(saved_hash, self.config_hash):
                if saved_hash is None:
                    logger.warning(
                        "Checkpoint %s 缺少 config_hash（旧格式或未启用指纹），"
                        "无法验证其基于当前配置，忽略旧断点，将全量重算",
                        self.checkpoint_path,
                    )
                elif "::" not in str(saved_hash) and "::" in self.config_hash:
                    logger.warning(
                        "Checkpoint %s 仅含旧格式本阶段指纹（saved=%s, current=%s），"
                        "无法验证上游配置，忽略旧断点，将全量重算",
                        self.checkpoint_path, saved_hash, self.config_hash,
                    )
                else:
                    logger.warning(
                        "Checkpoint %s 的配置指纹不一致（saved=%s, current=%s）："
                        "上游或本阶段配置已变更，忽略旧断点，将全量重算",
                        self.checkpoint_path, saved_hash, self.config_hash,
                    )
                self._done = set()
                return
            if isinstance(done, list):
                self._done = set(done)
            logger.info("Checkpoint 加载：%d 条已处理记录", len(self._done))
        except Exception as e:
            logger.warning("Checkpoint 加载失败 %s: %s", self.checkpoint_path, e)
            self._done = set()

    def is_done(self, record_id: str) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return record_id in self._done

    def mark_done(self, record_id: str) -> None:
        if not self.enabled:
            return
        should_save = False
        with self._lock:
            if record_id in self._done:
                return
            self._done.add(record_id)
            self._dirty = True
            self._marks_since_save += 1
            should_save = self._marks_since_save >= self.save_interval
        if should_save:
            self.save(force=False)

    def mark_done_batch(self, record_ids: Iterable[str]) -> None:
        # 批量 mark 时，只在最后触发一次 save 检查，避免每条记录都获取锁
        pending_save = False
        with self._lock:
            for rid in record_ids:
                if rid not in self._done:
                    self._done.add(rid)
                    self._dirty = True
                    self._marks_since_save += 1
                    if self._marks_since_save >= self.save_interval:
                        pending_save = True
        if pending_save:
            self.save(force=False)

    def save(self, force: bool = False) -> None:
        if not self.enabled or not self.checkpoint_path:
            return
        with self._save_lock:
            with self._lock:
                if not force and not self._dirty:
                    return
                # 百万级 record 下每次 save 全量排序开销大，直接快照为 list
                done_snapshot = list(self._done)
                marks_snapshot = self._marks_since_save
                self._dirty = False
                self._marks_since_save = 0

            # 在真正写 checkpoint 前，先触发所有 pre-save callback（如 flush manifest），
            # 保证 checkpoint 与 manifest 的一致性；任一 callback 失败则中止本次落盘并
            # 恢复 dirty 与 marks 计数（视同本次 save 未发生）——否则 checkpoint 会
            # 领先于 manifest，resume 时 is_done 跳过从未写入 manifest 的记录，造成静默丢数据
            try:
                for cb in self._pre_save_callbacks:
                    cb()
            except Exception as e:
                logger.warning("Checkpoint pre-save callback 失败，中止本次保存: %s", e)
                with self._lock:
                    self._dirty = True
                    self._marks_since_save += marks_snapshot
                return

            tmp_path = self.checkpoint_path + ".tmp"
            try:
                # checkpoint_path 可能是不含目录的裸文件名，此时 dirname 为空串，
                # os.makedirs("") 会抛 FileNotFoundError，需跳过
                ckpt_dir = os.path.dirname(self.checkpoint_path)
                if ckpt_dir:
                    os.makedirs(ckpt_dir, exist_ok=True)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "done": done_snapshot,
                            "count": len(done_snapshot),
                            "config_hash": self.config_hash,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                    # 掉电场景下仅 replace 不够：tmp 内容与目录项都必须 fsync，
                    # 否则断点可能丢失（已确认完成的记录被重算）
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace_durable(tmp_path, self.checkpoint_path)
                logger.debug("Checkpoint 保存：%d 条记录", len(done_snapshot))
            except Exception as e:
                logger.warning("Checkpoint 保存失败 %s: %s", self.checkpoint_path, e)
                with self._lock:
                    self._dirty = True
                    self._marks_since_save += marks_snapshot

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checkpoint_path": self.checkpoint_path,
            "done_count": len(self._done),
            "save_interval": self.save_interval,
        }
