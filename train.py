#!/usr/bin/env python3
"""Unified training entry point for SE, AEC, TSE, SS, and SSL tasks.

Auto-detects the task from the configuration file via the ``tasks.registry``
descriptors and routes to the appropriate training entry
(SE / AEC / TSE / SS / SSL). GAN is a training paradigm (SE ``PASEGANTrainer``)
rather than a separate task.

Usage::

    python train.py -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml
    python train.py -C configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml
"""

import argparse

from core.utils.config import load_merged_config

from core.trainer.entry import detect_task, set_spawn_start_method
from tasks.registry import load_training_entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-C", "--config", required=True)
    args, _ = p.parse_known_args()

    cfg = load_merged_config(args.config)
    task_info = detect_task(cfg)

    entry_cls = load_training_entry(task_info.task)
    entry = entry_cls(task_info.logger_name)
    entry.run(config=cfg, config_path=args.config)


if __name__ == "__main__":
    set_spawn_start_method()
    main()
