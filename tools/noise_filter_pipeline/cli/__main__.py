"""支持 ``PYTHONPATH=tools python -m noise_filter_pipeline.cli``。"""
from .main import main


if __name__ == "__main__":
    raise SystemExit(main())
