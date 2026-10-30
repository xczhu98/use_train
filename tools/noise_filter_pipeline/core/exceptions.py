"""Pipeline 共享异常定义。"""
from __future__ import annotations


class InfraAbortError(RuntimeError):
    """基础设施级中止信号：要求立即中止整个 stage。

    区别于内容级的单批推理失败（可标记 batch_failed 后继续），该异常表示
    基础设施已无法自愈（如 VAD 进程池连续多批损坏、AudioTag daemon 全部
    不可用），继续跑下去只会把全部数据静默降级为拒绝。各 batch 处理器必须
    用 ``except InfraAbortError: raise`` 将其透传，不得被 blanket
    ``except Exception`` 吞掉降级为逐批失败。
    """


class ContentLevelError(RuntimeError):
    """内容级错误：模型/daemon 正常返回错误结果（如批内单条损坏音频）。

    与基础设施失败（failed+retryable、不写 checkpoint）相反，内容级错误
    重试无意义——同一输入重跑必然再次失败。batch 处理器应将其整批标为
    终态 rejected（写 checkpoint），避免毒记录让 pipeline 每轮中止；
    与 C1/C3 的 UNPARSEABLE 终态拒绝语义一致。
    """
