"""Speech Separation inference APIs."""

from tasks.ss.inference.extractor import SSExtractor
from tasks.ss.inference.adapter import SSInferenceAdapter

__all__ = [
    "SSExtractor",
    "SSInferenceAdapter",
]
