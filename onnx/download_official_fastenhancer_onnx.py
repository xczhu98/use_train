#!/usr/bin/env python3
"""Download official FastEnhancer ONNX models from GitHub releases.

Usage:
    python onnx/download_official_fastenhancer_onnx.py [--variant B] [--dataset vctk-demand|dns|48khz]

Official releases:
  - VCTK-Demand 16kHz: https://github.com/aask1357/fastenhancer/releases/tag/onnx-vd-v1.0.0
  - DNS 16kHz:         https://github.com/aask1357/fastenhancer/releases/tag/onnx-dns-v1.0.0
  - 48kHz:             https://github.com/aask1357/fastenhancer/releases/tag/onnx-48khz-v1

Note: If direct download fails due to network restrictions, please download manually
and place the ONNX file under onnx/onnx_model/fastenhancer_{variant}_official.onnx.
HTTPS downloads use urllib's default system certificate and hostname verification.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.parse
from pathlib import Path


# 使用固定 asset digest/size 校验下载内容。
_OFFICIAL_RELEASES = {
    "vctk-demand": {
        "tag": "onnx-vd-v1.0.0",
        "waveform": {
            "t": ("9ed1b928d18fbead4db69bfbeb4ff99e3007f12abf9887d40fd41732674ccd0d", 125036),
            "b": ("abd453a3bae0e1100ae54b93bc15ae71f0663fc1346b85044eb341f5bce6530f", 413188),
            "s": ("e2d0e91bbfab4af1316bb2c41126a38c8b3cd015b93bb630d651af8fdbf7f2e8", 832775),
            "m": ("367059e724dd367c056dc906e9698ec5864c80c9a88e0597f7a2b0f81c506aaa", 2033628),
            "l": ("915d4ae3871c3271e75c194e72a3ff031d5593be4f6f56ccb78df82071ef0750", 4484713),
        },
        "spec": {
            "t": ("915a451f3b1ea8e98c20517c63b50943aa1d540624189f3106cc2e03d09634eb", 120579),
            "b": ("9dc4090af273c98fd66b2df2b439b61778f30aec84a3440432a6b68422adc26f", 408870),
            "s": ("457fa2b99f8c12ba7d59e66d0ebdaafbd14b221b9b31d47818afe494999fbd15", 829426),
            "m": ("fa3264db60f8f7d5df97670876ff660c7c152e41e483eaa8b9917d7d8ea811d4", 2029871),
            "l": ("47f7f63ff75bd8c35c1321f3c6ae2d31eea74f30a510289c11e7859d69c42e8f", 4481770),
        },
    },
    "dns": {
        "tag": "onnx-dns-v1.0.0",
        "waveform": {
            "t": ("a64663672b31bd1445502ee23328d291af13fb312e11bb574c010e7261a5ee7d", 125233),
            "b": ("e2d8758a91a5c7d59334fdd4b871900cbd311d6694d99d9e0a77662063e60434", 413220),
            "s": ("d7c937d6b475dbe89a6e6a69853b685fe62d629f4c96077559c556ac6f8aaa2d", 832813),
            "m": ("29da81b46ad42758d9f586fb1706902862ec3f0337d6bd49670c5565e42382fb", 2033752),
            "l": ("a211423e348707fba4952f2b868084a8d8880c6e8bd17333f4b1c0e87bd35b12", 4484966),
        },
        "spec": {
            "t": ("d51080705d024f798b6083b844f7459326d974150c78187d7403a2dfe8933565", 120579),
            "b": ("6d3f1e2779a09da4b313fef56a9648c55600320a223e997dc3cfb9909ba8683e", 408870),
            "s": ("e3ea546d7df58fc2ad40dfe8c3428d388801f04fc352690460840c4f7e66c6ca", 829426),
            "m": ("7ed8db52b80a45b260157bd335d14256f051c0edb86eed2cd6b144a3d785d703", 2029871),
            "l": ("ce9e0a2974fd0ba40fbed1c24663bfaf680601ba8c64faa4b99b9bd6cb0da900", 4481770),
        },
    },
    "48khz": {
        "tag": "onnx-48khz-v1",
        "waveform": {
            "t": ("1993a3f58ae95d959123b3d28779c2800e0f016d6f4e1177f1213144f301b89c", 151147),
            "b": ("70e23bba3d41e80d30ebc5eba39d9df64f0e0315f31c772022bb17576c4d96bf", 449559),
            "s": ("f04ece2beed330da367264c54cedded62f65a117fbde5c005d3a88fc796d0ba3", 875885),
            "m": ("c7da800810b583f4734d757c6e14d235f3eec81476121b595743e5866b66efa2", 2101876),
            "l": ("d7138309ec98266c668b2f86658fcc8f4e82ded67cd7fc8d6598534104b4bf89", 4593002),
        },
        "spec": {
            "t": ("ebc673b4346ecbb89d2dce13718dbac46db81c57b056e3cb0c79eb767405956a", 144919),
            "b": ("8df7941f5c9c81fdb79ccfeeb427ef7277d68821c1077eb940cc9893c0bbcac7", 444150),
            "s": ("a47a4aea48c7bfd053eb83b6b986bbaa518ad26c4979af3736d8ef963cefd818", 872039),
            "m": ("12a5fe6a28bd32ca02d2838dbed3c33954c1ca35e1e6440f47121c6b79aec859", 2097348),
            "l": ("0817aa8ea292915428b9dfbe08cd39496efe7162e542b6ffe5755c5100687991", 4589899),
        },
    },
}


def _validated_content_length(headers) -> int | None:
    raw = headers.get("Content-Length") if headers is not None else None
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid Content-Length header: {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"Invalid Content-Length header: {raw!r}")
    return value


def _validate_output_dir(output_dir: Path) -> Path:
    """限制下载目标为共享 publisher 可安全发布的 release-root 子目录。"""
    from onnx.verify.verify_common import onnx_release_root

    release_root_path = onnx_release_root()
    try:
        release_root = release_root_path.resolve(strict=True)
        candidate = Path(os.path.abspath(output_dir.expanduser()))
        relative = candidate.relative_to(release_root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "--output-dir must be under the ONNX release root "
            f"{release_root_path}: {output_dir}"
        ) from exc

    current = release_root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(
                "--output-dir must not contain symlinks under the ONNX release root: "
                f"{output_dir}"
            )
        if current.exists() and not current.is_dir():
            raise ValueError(f"--output-dir is not a directory: {output_dir}")
    return candidate


def _download_atomic(
    url: str,
    out_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> dict:
    """下载固定字节并交由共享锁/事务 publisher 发布。"""
    import urllib.request

    from onnx.verify.verify_common import publish_verified_onnx

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a pinned lowercase SHA-256 digest")
    if expected_size_bytes < 1:
        raise ValueError("expected_size_bytes must be positive")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            final_url = response.geturl()
            if urllib.parse.urlsplit(final_url).scheme.lower() != "https":
                raise RuntimeError(f"Download redirected to non-HTTPS URL: {final_url}")
            content_length = _validated_content_length(response.headers)
            if content_length is not None and content_length != expected_size_bytes:
                raise RuntimeError(
                    "Pinned size mismatch before download: "
                    f"expected {expected_size_bytes}, server declared {content_length}"
                )
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent
            )
            temporary_path = Path(temporary_name)
            digest = hashlib.sha256()
            total = 0
            with os.fdopen(fd, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        if total != expected_size_bytes:
            raise RuntimeError(
                f"Pinned size mismatch: expected {expected_size_bytes}, received {total}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        result = publish_verified_onnx(
            temporary_path,
            out_path,
            all_pass=True,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        temporary_path = None
        if result is None:
            raise RuntimeError("official ONNX publisher returned no result")
        return result
    except Exception as download_error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"{download_error}; download_cleanup=retained: {cleanup_error}"
                ) from download_error
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Download official FastEnhancer ONNX")
    ap.add_argument("--variant", type=str, default="B", choices=["T", "B", "S", "M", "L"])
    ap.add_argument("--dataset", type=str, default="vctk-demand",
                    choices=["vctk-demand", "dns", "48khz"],
                    help="Training dataset used for the released model")
    ap.add_argument("--spec", action="store_true",
                    help="Download spec-in/spec-out version (_spec.onnx) instead of waveform-in/waveform-out")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "onnx_model",
        help="ONNX release-root 内的目标目录（可使用其非 symlink 子目录）",
    )
    ns = ap.parse_args()
    try:
        ns.output_dir = _validate_output_dir(ns.output_dir)
    except ValueError as exc:
        ap.error(str(exc))

    release = _OFFICIAL_RELEASES[ns.dataset]
    variant = ns.variant.lower()
    mode = "spec" if ns.spec else "waveform"
    expected_sha256, expected_size_bytes = release[mode][variant]
    tag = release["tag"]
    if ns.spec and ns.dataset in {"vctk-demand", "dns"}:
        filename = f"fastenhancer_{variant}.spec.onnx"
    else:
        suffix = "_spec" if ns.spec else ""
        filename = f"fastenhancer_{variant}{suffix}.onnx"
    url = f"https://github.com/aask1357/fastenhancer/releases/download/{tag}/{filename}"

    suffix = "_spec" if ns.spec else ""
    out_path = ns.output_dir / f"fastenhancer_{variant}_official{suffix}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {url}")
    print(f"Destination: {out_path}")

    try:
        result = _download_atomic(
            url,
            out_path,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        cleanup = {
            key: value for key, value in result.items() if key.endswith("_cleanup")
        }
        cleanup_text = f" cleanup={cleanup}" if cleanup else " cleanup=complete"
        print(
            f"Success: {result['size_bytes']} bytes saved; "
            f"sha256={result['sha256']};{cleanup_text}"
        )
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nPlease download manually from:")
        print(f"  {url}")
        print(f"And place it at: {out_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
