"""C1 Qwen 单事件处理器的轻量级契约测试。"""
import pytest

from noise_filter_pipeline.core.manifest_io import config_hash
from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
from noise_filter_pipeline.processors.qwen_single_event import (
    QwenSingleEventModel,
    QwenSingleEventProcessor,
)


pytestmark = pytest.mark.unit


class _FakeInputs(dict):
    """模拟 transformers BatchFeature 的链式设备转换。"""

    def to(self, _device_or_dtype):
        return self


class _FakeProcessor:
    def apply_chat_template(self, conversations, **kwargs):
        return ["formatted"] * len(conversations)

    def __call__(self, **kwargs):
        return _FakeInputs(input_ids=[1])

    def batch_decode(self, text_ids, **kwargs):
        return ["<|im_start|>assistant\nNoise"]


def test_qwen_batch_query_raw_passes_thinker_token_cap():
    """C1 生成调用必须显式限制 thinker 的最大新 token 数。"""
    captured = {}

    class _FakeModel:
        device = "cpu"
        dtype = "float32"

        def generate(self, **kwargs):
            captured.update(kwargs)
            return [[1]]

    model = QwenSingleEventModel("/fake/qwen", max_new_tokens=5)
    model._model = _FakeModel()
    model._processor = _FakeProcessor()
    model._process_mm_info = lambda conversations, **kwargs: ([], [], [])

    assert model.batch_query_raw([{"audio_path": "a.wav", "prompt": "p"}]) == [
        "<|im_start|>assistant\nNoise"
    ]
    assert captured["thinker_max_new_tokens"] == 5
    assert captured["return_audio"] is False


def test_qwen_max_new_tokens_default_and_fingerprint_identity():
    """生成上限默认 8，切换上限必须使 C1 checkpoint 与 cache 失效。"""
    default_proc = QwenSingleEventProcessor({"dry_run": True})
    tuned_proc = QwenSingleEventProcessor({"dry_run": True, "max_new_tokens": 3})

    assert default_proc.max_new_tokens == 8
    assert tuned_proc.max_new_tokens == 3
    assert default_proc.config["max_new_tokens"] == 8
    assert tuned_proc.config["max_new_tokens"] == 3
    assert config_hash(default_proc.config) != config_hash(tuned_proc.config)
    assert default_proc.cache.key_salt == "max_new_tokens=8"
    assert tuned_proc.cache.key_salt == "max_new_tokens=3"


def test_pipeline_config_accepts_c1_max_new_tokens():
    """Stage C1 exact schema 应接受 max_new_tokens 配置项。"""
    config = UnifiedPipelineConfig.from_dict(
        {"stage_c": {"c1": {"max_new_tokens": 3}}},
        active_stages=(),
    )
    assert config.stage_c["c1"]["max_new_tokens"] == 3
