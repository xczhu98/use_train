#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 noise_filter_pipeline 的 nemo_onnx VAD 准备 ONNX 模型文件。
推理时只需 onnxruntime + torch/torchaudio，无需安装 nemo_toolkit。

用法:
  python tools/noise_filter_pipeline/scripts/download_nemo_vad_onnx.py

逻辑（按顺序）:
  1. 若 third_party_scripts/NeMo-VAD-onnx/ 下已有可加载的 ONNX 文件，则无需任何操作。
  2. 若设置了环境变量 NEMO_VAD_ONNX_URL，则仅从 HTTPS URL 下载并校验 SHA-256。
  3. 若本机已安装 nemo_toolkit，则先导出到临时文件并校验，再原子发布。
  4. 否则仅提示：请手动将 ONNX 文件放到该目录。
"""

import argparse
import hashlib
import os
import re
import stat
import sys
import tempfile
import urllib.request
from urllib.parse import urlsplit


DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _get_third_party_nemo_onnx_dir():
    """与 processors/vad/bootstrap.py 中路径一致：use_train/third_party_scripts/NeMo-VAD-onnx"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.dirname(current_dir)
    tools_dir = os.path.dirname(pipeline_root)
    use_train_root = os.path.dirname(tools_dir)
    third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")
    return os.path.join(third_party, "NeMo-VAD-onnx")


def _validate_onnx_file(path):
    """严格校验 ONNX 产物是非空普通文件且能被现有运行时加载。"""
    path = os.fspath(path)
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"ONNX 文件不可访问: {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"ONNX 路径不是普通文件: {path}")
    if file_stat.st_size <= 0:
        raise ValueError(f"ONNX 文件为空: {path}")

    onnx_error = None
    try:
        import onnx
    except Exception as exc:  # 可选依赖缺失或损坏时转交 onnxruntime
        onnx = None
        onnx_error = exc
    if onnx is not None:
        checker = getattr(onnx, "checker", None)
        if checker is not None and hasattr(checker, "check_model"):
            try:
                model = onnx.load(path)
                checker.check_model(model)
            except Exception as exc:
                raise ValueError(f"ONNX checker 拒绝文件: {path}: {exc}") from exc
            return True
        onnx_error = ImportError("onnx.checker.check_model 不可用")

    try:
        import onnxruntime
    except Exception as exc:
        detail = f"；onnx 校验错误: {onnx_error}" if onnx_error else ""
        raise RuntimeError(f"缺少可用的 ONNX 校验器，已 fail-closed{detail}") from exc
    try:
        onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError(f"ONNX 文件无法由 onnxruntime 加载: {path}: {exc}") from exc
    return True


def _existing_onnx_path(target_dir):
    for name in ("vad_marblenet.onnx", "vad.onnx"):
        path = os.path.join(target_dir, name)
        try:
            _validate_onnx_file(path)
        except (OSError, RuntimeError, ValueError) as exc:
            if os.path.lexists(path):
                print(f"忽略无效 ONNX 文件 {path}: {exc}")
            continue
        return path
    return None


def _validate_download_url(url):
    """校验 URL，并拒绝初始地址及所有重定向中的非 HTTPS 地址。"""
    try:
        parsed = urlsplit(url)
        # 访问 port 以便同时拒绝非法端口等 malformed URL。
        parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"NEMO_VAD_ONNX_URL 无效: {exc}") from exc
    if parsed.scheme.lower() != "https":
        raise ValueError("NEMO_VAD_ONNX_URL 必须使用 https scheme")
    if not hostname:
        raise ValueError("NEMO_VAD_ONNX_URL 必须包含 hostname")


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """确保 urllib 不会跟随 HTTPS 到 HTTP 的降级重定向。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _trusted_download_sha256():
    value = os.environ.get("NEMO_VAD_ONNX_SHA256", "").strip()
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError("远端下载必须设置有效的 NEMO_VAD_ONNX_SHA256（64 位十六进制 SHA-256）")
    return value.lower()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_to_temp(url, tmp_path):
    request = urllib.request.Request(url)
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler())
    with opener.open(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise ValueError("下载响应的 Content-Length 无效") from exc
            if declared_size > MAX_DOWNLOAD_BYTES:
                raise ValueError("下载文件超过允许的最大大小")

        digest = hashlib.sha256()
        size = 0
        with open(tmp_path, "wb") as file_obj:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("下载文件超过允许的最大大小")
                file_obj.write(chunk)
                digest.update(chunk)
    return digest.hexdigest()


def _try_download(target_dir, out_path):
    url = os.environ.get("NEMO_VAD_ONNX_URL", "").strip()
    if not url:
        return False
    _validate_download_url(url)
    trusted_sha256 = _trusted_download_sha256()
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(out_path)}.", suffix=".download.tmp", dir=target_dir
        )
        os.close(fd)
        print(f"正在从 {url} 下载...")
        actual_sha256 = _download_to_temp(url, tmp_path)
        if actual_sha256 != trusted_sha256:
            raise ValueError("下载文件 SHA-256 与 NEMO_VAD_ONNX_SHA256 不匹配")
        _validate_onnx_file(tmp_path)
        os.replace(tmp_path, out_path)
        tmp_path = None
        _validate_onnx_file(out_path)
        print(f"已下载: {out_path}")
        return True
    except Exception as exc:
        print(f"下载失败: {exc}")
        return False
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _try_export_with_nemo(target_dir, out_path):
    try:
        from nemo.collections.asr.models import EncDecClassificationModel
    except ImportError:
        return False

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(out_path)}.", suffix=".export.tmp", dir=target_dir
        )
        os.close(fd)
        try:
            model = EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet")
        except Exception:
            model = EncDecClassificationModel.from_pretrained(model_name="vad_telephony_marblenet")
        model.eval()
        model.export(tmp_path)
        _validate_onnx_file(tmp_path)
        os.replace(tmp_path, out_path)
        tmp_path = None
        _validate_onnx_file(out_path)
        print(f"已导出: {out_path}")
        return True
    except Exception as exc:
        print(f"导出 NeMo VAD ONNX 失败: {exc}")
        return False
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)


def main(argv=None):
    if argv is not None:
        _parse_args(argv)
    target_dir = _get_third_party_nemo_onnx_dir()
    os.makedirs(target_dir, exist_ok=True)

    existing = _existing_onnx_path(target_dir)
    if existing:
        print(f"已存在 ONNX 文件: {existing}，无需操作。")
        return 0

    print(f"目标目录: {target_dir}")
    out_path = os.path.join(target_dir, "vad_marblenet.onnx")

    download_url_configured = bool(os.environ.get("NEMO_VAD_ONNX_URL", "").strip())
    if _try_download(target_dir, out_path):
        return 0
    if download_url_configured:
        print("已显式设置 NEMO_VAD_ONNX_URL，但下载或校验失败；不会回退到本地 NeMo 导出。")
        return 1

    if _try_export_with_nemo(target_dir, out_path):
        return 0

    print("未找到可用 ONNX 文件，且远端下载和 NeMo 导出均失败。")
    print("请将 vad_marblenet.onnx 或 vad.onnx 放到以下目录后即可使用 nemo_onnx VAD：")
    print(f"  {target_dir}")
    print("获取方式：从已安装 NeMo 的机器运行本脚本导出后拷贝，或设置 NEMO_VAD_ONNX_URL 与 NEMO_VAD_ONNX_SHA256 后重新运行本脚本。")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
