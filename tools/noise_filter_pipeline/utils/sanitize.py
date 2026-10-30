"""路径/文件名清洗工具（供 scripts/rename_paths_replace_spaces.py 等使用）。"""
import os
import re

# 路径/目录名安全：仅保留 Unicode 字母数字、下划线、连字符；空格、逗号、句号等一律为 _
_UNSAFE_PATH_SEGMENT = re.compile(r"[^\w\-]+", re.UNICODE)


def _sanitize_path_token(part: str, empty_fallback: str) -> str:
    if not isinstance(part, str):
        return empty_fallback
    t = part.replace(os.sep, "_").replace("..", "_")
    t = _UNSAFE_PATH_SEGMENT.sub("_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or empty_fallback


def sanitize_basename_for_rename(name: str, is_file: bool) -> str:
    """重命名用：目录名整段清洗；文件名只洗 stem，保留扩展名。"""
    if not name:
        return "unnamed" if is_file else "unknown"
    if is_file:
        stem, suf = os.path.splitext(name)
        return _sanitize_path_token(stem, "unnamed") + suf
    return _sanitize_path_token(name, "unknown")
