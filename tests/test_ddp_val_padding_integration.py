"""真实 2 卡 DDP 集成测试：val 尾部对齐复制批"跑而不计"语义（Fix C 回归）。

与 test_val_ddp_padding.py 的打桩单测互补：本测试用当前解释器的
``torch.distributed.run`` 起 2 个真实 NCCL 进程，验证端到端行为：

1. 各 rank 批数不等时 val 循环**不挂死**（broadcast_buffers=True 下每次
   DDP forward 都有 buffer 广播 collective，复制批的 forward 必须照跑）；
2. 复制批的贡献确实被剔除：sum/sum allreduce 后的 val mean 等于"每个真实
   样本恰计一次"的期望值；
3. 对照组（不剔除）均值确实不同，证明测试场景真的触发了复制路径。

运行方式：pytest 内 subprocess 调 ``python -m torch.distributed.run
--nproc_per_node=2``；无 GPU 或只有单卡时自动 skip。
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

_DDP_SCRIPT = textwrap.dedent('''
    import os
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    import torch.nn as nn

    from core.data.sampler import GroupedBatchSampler
    from core.trainer.metrics_trainer import MetricsCapableTrainer


    class _FakeDataset:
        def __init__(self, srs, lengths):
            self._srs = list(srs)
            self._lengths = list(lengths)

        def get_srs(self):
            return list(self._srs)

        def get_source_length(self):
            return list(self._lengths)


    class _ToyModel(nn.Module):
        """带 BatchNorm buffer 的最小模型：确保每次 DDP forward 都有
        buffer 广播 collective（broadcast_buffers=True 默认）。"""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 4, 3, padding=1)
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x):
            return self.bn(self.conv(x)).mean()


    def main():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # 49 样本、单 SR：stride 分片后 rank0=25 / rank1=24，
        # batch=4 → 7 vs 6 批，rank1 必然触发尾部复制对齐。
        n = 49
        ds = _FakeDataset([16000] * n, [100 + i for i in range(n)])
        sampler = GroupedBatchSampler(
            ds, batch_size=4, rank=rank, world_size=2, seed=3,
            bucket_size_mult=100, local_rank=rank,
        )
        sampler.set_epoch(0)

        model = nn.parallel.DistributedDataParallel(
            _ToyModel().to(device), device_ids=[rank]
        )
        model.eval()

        # 每个迭代的"loss"取其 1-based 序号：剔除语义可用闭式期望值精确断言，
        # 不依赖模型数值。rank0 真实批 7 个、rank1 真实批 6 个；
        # rank1 的第 7 次迭代是复制批（forward 照跑，贡献应剔除）。
        n0, n1 = 7, 6
        total = 0.0
        cnt = 0
        total_with_dup = 0.0
        cnt_with_dup = 0
        batches = list(iter(sampler))
        real = sampler._last_iter_real_len
        assert real == (n0 if rank == 0 else n1), (rank, real, len(batches))
        assert len(batches) == n0  # 两 rank 均补齐到 target=7

        fake_trainer = SimpleNamespace(
            val_loader=SimpleNamespace(batch_sampler=sampler)
        )
        is_pad = MetricsCapableTrainer._val_iter_is_ddp_padding

        with torch.inference_mode():
            for n_loops, _batch in enumerate(batches, start=1):
                # forward 必须照跑（buffer 广播同步），哪怕该迭代最终不计入
                x = torch.randn(2, 1, 32, device=device)
                _ = model(x)
                val = float(n_loops)
                total_with_dup += val
                cnt_with_dup += 1
                if not is_pad(fake_trainer, n_loops):
                    total += val
                    cnt += 1

        t_sum = torch.tensor([total, total_with_dup], device=device)
        t_cnt = torch.tensor([float(cnt), float(cnt_with_dup)], device=device)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

        # 剔除后：mean = (sum(1..7) + sum(1..6)) / 13 = 49 / 13
        expect_dedup = (n0 * (n0 + 1) / 2 + n1 * (n1 + 1) / 2) / (n0 + n1)
        # 不剔除：rank1 多计一次迭代值 7 → (28 + 28) / 14 = 4.0
        expect_with_dup = (
            n0 * (n0 + 1) / 2 + n1 * (n1 + 1) / 2 + (n1 + 1)
        ) / (2 * n0)

        got_dedup = (t_sum[0] / t_cnt[0]).item()
        got_with_dup = (t_sum[1] / t_cnt[1]).item()
        assert abs(got_dedup - expect_dedup) < 1e-6, (got_dedup, expect_dedup)
        assert abs(got_with_dup - expect_with_dup) < 1e-6, (got_with_dup, expect_with_dup)
        # 对照：不剔除时均值确实不同（复制批真的进了指标）
        assert abs(got_with_dup - got_dedup) > 1e-3

        # 双 rank 都走到这里 → 无挂死
        dist.barrier()
        if rank == 0:
            print("DDP_VAL_PADDING_INTEGRATION_OK")
        dist.destroy_process_group()


    if __name__ == "__main__":
        main()
''')


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="需要 >= 2 张 GPU 跑真实 NCCL DDP",
)
def test_ddp_val_padding_end_to_end(tmp_path):
    torchrun = [sys.executable, "-m", "torch.distributed.run"]
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "ddp_val_padding_integration.py"
    script.write_text(_DDP_SCRIPT)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [*torchrun, "--standalone", "--nproc_per_node=2", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(repo),
    )
    assert proc.returncode == 0, (
        f"torch.distributed.run 失败 (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "DDP_VAL_PADDING_INTEGRATION_OK" in proc.stdout

pytestmark = [pytest.mark.integration, pytest.mark.gpu]
