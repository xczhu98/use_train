"""
LLM 推理结果磁盘缓存。

以 (audio_path, prompt_text, model_name) 为 key 持久化结果，并按模型 identity
隔离 namespace，支持断点续跑、去重以及分布式场景下的缓存共享。缓存项用 SHA256
哈希作为文件名，避免特殊字符。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMResultCache:
    """LLM 结果磁盘缓存。

    key 含 (model_name, audio_path, 文件 size+mtime, prompt)。namespace 绑定已验证的
    Qwen model SHA；namespace 变化时使用独立目录，避免同路径模型替换后复用旧结果。
    旧文件仍占用磁盘，如需清理可直接删除 cache_dir 后重建（缓存随时可重算，
    无数据风险）。
    """

    def __init__(
        self,
        cache_dir: Optional[str],
        model_name: str = "default",
        namespace: str = "unset:qwen_model_sha256",
        key_salt: str = "",
    ):
        self.model_name = model_name or "default"
        self.namespace = namespace or "unset:qwen_model_sha256"
        self.key_salt = key_salt
        if (
            self.namespace in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.:-]+", self.namespace) is None
        ):
            raise ValueError("LLM cache namespace 必须是安全的单一路径组件")
        self.cache_dir: Optional[Path] = None
        self._cache_root: Optional[Path] = None
        if cache_dir:
            configured_root = Path(cache_dir).expanduser().absolute()
            configured_root.mkdir(parents=True, exist_ok=True)
            self._cache_root = Path(os.path.realpath(configured_root))
            target = self._cache_root / self.namespace
            target.mkdir(parents=False, exist_ok=True)
            if target.resolve().parent != self._cache_root:
                raise ValueError("LLM cache namespace 逃逸 cache_dir")
            self._require_safe_directory(target, purpose="namespace")
            self.cache_dir = target

    @staticmethod
    def _require_safe_directory(path: Path, *, purpose: str) -> None:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"LLM cache {purpose} 必须是实体目录: {path}")

    def _key(self, audio_path: str, prompt: str, model_name: str) -> str:
        # key 中加入文件 size+mtime 指纹：同一路径的音频被重新生成（如换分片
        # 参数重跑 Stage B，输出路径确定性不变）时，旧结论不得被复用
        try:
            st = os.stat(audio_path)
            fingerprint = f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            fingerprint = "missing"
        text = (
            f"{model_name}::{self.key_salt}::{audio_path}::"
            f"{fingerprint}::{prompt}"
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str, *, create_bucket: bool = False) -> Optional[Path]:
        if not self.cache_dir:
            return None
        # 按 key 前两位分桶（256 个子目录），避免百万级条目下单目录 inode 爆炸
        bucket = self.cache_dir / key[:2]
        if create_bucket:
            bucket.mkdir(parents=False, exist_ok=True)
        if bucket.exists() or os.path.lexists(bucket):
            self._require_safe_directory(bucket, purpose="bucket")
            if bucket.resolve().parent != self.cache_dir.resolve():
                raise ValueError("LLM cache bucket 逃逸 namespace")
        return bucket / f"{key}.json"

    def _open_bucket_fd(self, key: str, *, create: bool) -> Optional[int]:
        """经 namespace dirfd 打开桶目录，拒绝 symlink 与路径替换逃逸。"""
        if not self.cache_dir:
            return None
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        namespace_fd = os.open(self.cache_dir, flags)
        try:
            bucket_name = key[:2]
            if create:
                try:
                    os.mkdir(bucket_name, dir_fd=namespace_fd)
                except FileExistsError:
                    pass
            try:
                return os.open(bucket_name, flags, dir_fd=namespace_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise ValueError(
                    f"LLM cache bucket 必须是实体目录: {self.cache_dir / bucket_name}"
                ) from exc
        finally:
            os.close(namespace_fd)

    def get(self, audio_path: str, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        key = self._key(audio_path, prompt, self.model_name)
        cache_path = self._cache_path(key)
        bucket_fd = self._open_bucket_fd(key, create=False)
        if bucket_fd is None:
            return None
        try:
            try:
                fd = os.open(
                    f"{key}.json",
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=bucket_fd,
                )
            except FileNotFoundError:
                return None
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                os.close(fd)
                raise ValueError(f"LLM cache entry 必须是 regular file: {cache_path}")
            try:
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("读取 LLM 缓存失败 %s: %s", cache_path, e)
                return None
        finally:
            os.close(bucket_fd)

    def set(
        self,
        audio_path: str,
        prompt: str,
        result: Dict[str, Any],
    ) -> None:
        if not self.cache_dir:
            return
        key = self._key(audio_path, prompt, self.model_name)
        cache_path = self._cache_path(key)
        bucket_fd = self._open_bucket_fd(key, create=True)
        if not cache_path or bucket_fd is None:
            return
        final_name = f"{key}.json"
        tmp_name = f"{final_name}.{os.getpid()}.tmp"
        try:
            # 独占 tmp + rename 原子写，避免进程中断留下半截 JSON；文件名严格
            # 遵循 cleanup 的 <key>.json.<pid>.tmp 契约，残留可被统一回收。
            # 所有 I/O 均绑定已打开 bucket dirfd，路径替换不会重定向到外部。
            fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=bucket_fd,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            os.rename(tmp_name, final_name, src_dir_fd=bucket_fd, dst_dir_fd=bucket_fd)
        except Exception as e:
            try:
                os.unlink(tmp_name, dir_fd=bucket_fd)
            except OSError:
                pass
            logger.warning("写入 LLM 缓存失败 %s: %s", cache_path, e)
        finally:
            os.close(bucket_fd)

    def has(self, audio_path: str, prompt: str) -> bool:
        if not self.cache_dir:
            return False
        key = self._key(audio_path, prompt, self.model_name)
        bucket_fd = self._open_bucket_fd(key, create=False)
        if bucket_fd is None:
            return False
        try:
            info = os.stat(f"{key}.json", dir_fd=bucket_fd, follow_symlinks=False)
            return stat.S_ISREG(info.st_mode)
        except FileNotFoundError:
            return False
        finally:
            os.close(bucket_fd)
