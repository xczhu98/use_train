"""Shared data-processing toolkit for use_train pipelines.

This package holds utilities that are reused across:
  - tools/prepare_data_tools/
  - tools/audio_filter_pipeline/
  - tools/noise_filter_pipeline/

Modules:
  - audio_io: load/save/resample/split/detect silence and related helpers
  - scp_io: SCP / list / JSON index writers and readers

Existing per-pipeline modules are kept as thin compatibility shims so imports
keep working during migration.
"""
