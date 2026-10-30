"""RelPositionalEncoding inference cache 契约测试。"""

import pytest
import torch

from tasks.se.models.discriminative.conformer import RelPositionalEncoding


pytestmark = [pytest.mark.unit]


def test_rel_positional_cache_is_nonpersistent_buffer_and_tracks_length_dtype():
    encoding = RelPositionalEncoding(d_model=8, dropout=0.0)

    _, pos_short = encoding(torch.zeros(2, 3, 8, dtype=torch.float32))
    assert pos_short.shape == (1, 3, 8)
    assert encoding.pe.shape == (1, 3, 8)
    assert encoding.pe.dtype == torch.float32
    assert "pe" not in encoding.state_dict()
    assert dict(encoding.named_buffers())["pe"] is encoding.pe

    _, pos_long = encoding(torch.zeros(1, 7, 8, dtype=torch.float64))
    assert pos_long.shape == (1, 7, 8)
    assert encoding.pe.shape == (1, 7, 8)
    assert encoding.pe.dtype == torch.float64

    _, pos_reused = encoding(torch.zeros(1, 2, 8, dtype=torch.float32))
    assert pos_reused.shape == (1, 2, 8)
    assert encoding.pe.shape == (1, 7, 8)
    assert encoding.pe.dtype == torch.float32
    assert pos_reused.dtype == torch.float32
    assert "pe" not in encoding.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_rel_positional_cache_migrates_to_cuda_input():
    encoding = RelPositionalEncoding(d_model=8, dropout=0.0)
    encoding(torch.zeros(1, 4, 8))

    x = torch.zeros(1, 2, 8, device="cuda", dtype=torch.float16)
    _, pos_emb = encoding(x)

    assert encoding.pe.device == x.device
    assert encoding.pe.dtype == x.dtype
    assert pos_emb.device == x.device
    assert pos_emb.dtype == x.dtype
