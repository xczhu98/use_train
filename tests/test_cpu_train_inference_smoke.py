"""最小 CPU train -> checkpoint -> evaluate/inference 闭环。"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
import torch

from core.inference import FileInferenceItem, PreparedOutput
from core.utils.artifact_identity import bind_source_files
from core.utils.safe_load import safe_torch_load
from evaluate.inference_components import run_adapter_inference

pytestmark = pytest.mark.unit


def _write_silence(path: Path, sample_rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * 16)


class _TinyAudioAdapter:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def infer(self, item: FileInferenceItem) -> PreparedOutput:
        value = float(self.model.weight.detach().cpu().item())

        def writer(destination: Path) -> None:
            with wave.open(str(destination), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                sample = max(-32768, min(32767, round(value * 1000)))
                handle.writeframes(int(sample).to_bytes(2, "little", signed=True) * 16)

        return PreparedOutput(uid=item.uid, writer=writer)


def test_cpu_train_checkpoint_evaluate_and_reuse(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(torch.ones(1, 1)) - torch.ones(1, 1)).square().mean()
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    loaded = safe_torch_load(checkpoint)
    reloaded = torch.nn.Linear(1, 1, bias=False)
    reloaded.load_state_dict(loaded["model"], strict=True)

    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\n", encoding="utf-8")
    noisy = tmp_path / "noisy.wav"
    _write_silence(noisy)
    output_dir = tmp_path / "outputs"
    item = FileInferenceItem(
        uid="utt-1",
        inputs={"noisy": noisy},
        output=output_dir / "utt-1.wav",
    )
    binding = bind_source_files(
        {"checkpoint": checkpoint, "config": config},
        parameters={"task": "tiny-se"},
    )

    first = run_adapter_inference(
        adapter=_TinyAudioAdapter(reloaded),
        config=config,
        output_dir=output_dir,
        items=[item],
        artifact_fingerprint=binding,
        checkpoint=checkpoint,
    )
    assert first.status == "completed"
    assert first.results[0].status == "ok"
    assert item.output.exists()

    second = run_adapter_inference(
        adapter=_TinyAudioAdapter(reloaded),
        config=config,
        output_dir=output_dir,
        items=[item],
        artifact_fingerprint=binding,
        checkpoint=checkpoint,
        skip_existing=True,
    )
    assert second.results[0].status == "skipped"
