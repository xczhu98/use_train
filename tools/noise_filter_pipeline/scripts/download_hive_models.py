#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 noise_filter_pipeline Stage C/D 下载 Hive 依赖模型。

下载目标：
  - Apollo Universal 超分模型
      third_party_scripts/Apollo/apollo_model_uni.ckpt
      third_party_scripts/Apollo/config_apollo_uni.yaml
  - icefall AudioTag 模型（AudioSet）
      third_party_scripts/icefall_audiotaq/exp/pretrained.pt
      third_party_scripts/icefall_audiotaq/data/class_labels_indices.csv

用法:
  python tools/noise_filter_pipeline/scripts/download_hive_models.py
"""

import os
import subprocess
import sys
import urllib.request


# 注意：当前 C1/C3 loader 使用 Qwen2_5Omni 系列类，因此这里固定为
# Qwen2.5-Omni-7B；当前 loader 不支持 Qwen3-Omni。
QWEN_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
QWEN_TARGET_DIR = "Qwen2.5-Omni-7B"

MODELS = {
    "Apollo": {
        "dir": "Apollo",
        "files": [
            {
                "url": "https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/apollo_model_uni.ckpt",
                "name": "apollo_model_uni.ckpt",
            },
            {
                "url": "https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/config_apollo_uni.yaml",
                "name": "config_apollo_uni.yaml",
            },
        ],
    },
    "icefall_audiotaq": {
        "dir": "icefall_audiotaq",
        "files": [
            {
                "url": "https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12/resolve/main/exp/pretrained.pt",
                "name": "exp/pretrained.pt",
            },
            {
                "url": "https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12/resolve/main/data/class_labels_indices.csv",
                "name": "data/class_labels_indices.csv",
            },
        ],
    },
}


def _download_qwen(third_party: str) -> bool:
    target = os.path.join(third_party, QWEN_TARGET_DIR)
    if os.path.isdir(target) and any(os.scandir(target)):
        print(f"  已存在，跳过: {target}")
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"  下载 Qwen 需要 huggingface_hub: {e}")
        print(f"  请手动执行: huggingface-cli download {QWEN_MODEL_ID} --local-dir " + target)
        return False
    print(f"  开始下载 {QWEN_MODEL_ID}，目标: {target}")
    print("  注意：模型体积约 15-20GB，下载可能需要较长时间")
    try:
        snapshot_download(QWEN_MODEL_ID, local_dir=target)
        print(f"  Qwen 模型已下载到: {target}")
        return True
    except Exception as e:
        print(f"  Qwen 下载失败: {e}")
        return False


def _get_third_party_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.dirname(current_dir)
    tools_dir = os.path.dirname(pipeline_root)
    use_train_root = os.path.dirname(tools_dir)
    return os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")


ICEFALL_REPO_URL = "https://github.com/k2-fsa/icefall.git"
ICEFALL_REPO_DIR = "icefall"


def _clone_icefall(third_party: str) -> bool:
    target = os.path.join(third_party, ICEFALL_REPO_DIR)
    if os.path.isdir(os.path.join(target, ".git")):
        print(f"  icefall 已存在，跳过: {target}")
        return True
    print(f"  克隆 icefall -> {target}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", ICEFALL_REPO_URL, target],
            check=True,
        )
        return True
    except Exception as e:
        print(f"    克隆失败: {e}")
        return False


def _download(url: str, dest: str) -> bool:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  已存在，跳过: {dest}")
        return True
    print(f"  下载: {url}")
    print(f"    -> {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # 先下载到临时文件，成功后原子替换：部分下载不得被下次运行当作
    # "已存在"的完整产物跳过；失败时清理临时残留
    tmp_dest = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_dest)
        os.replace(tmp_dest, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    完成，大小: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"    下载失败: {e}")
        try:
            os.remove(tmp_dest)
        except OSError:
            pass
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载 Hive 数据管道依赖模型")
    parser.add_argument(
        "--qwen",
        action="store_true",
        help=f"同时下载 Qwen2.5-Omni 模型（{QWEN_MODEL_ID}，约 15-20GB）",
    )
    args = parser.parse_args()

    third_party = _get_third_party_dir()
    print(f"第三方模型根目录: {third_party}")

    all_ok = True
    for model_name, model_info in MODELS.items():
        print(f"\n[{model_name}]")
        model_dir = os.path.join(third_party, model_info["dir"])
        for item in model_info["files"]:
            dest = os.path.join(model_dir, item["name"])
            if not _download(item["url"], dest):
                all_ok = False

    if args.qwen:
        print(f"\n[Qwen2.5-Omni]")
        if not _download_qwen(third_party):
            all_ok = False
    else:
        print(f"\n[Qwen2.5-Omni]")
        print("  跳过（默认不下载，体积约 15-20GB）。如需下载请加 --qwen")

    print("\n[icefall]")
    if not _clone_icefall(third_party):
        all_ok = False

    if all_ok:
        print("\n所有模型与 icefall 代码已就绪")
    else:
        print("\n部分资源下载失败，请检查网络或手动下载")
        sys.exit(1)


if __name__ == "__main__":
    main()
