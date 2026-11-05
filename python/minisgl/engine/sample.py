from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)

# 将预处理好的概率张量和截断参数（温度、Top-K、Top-P）传递给 FlashInfer
def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling
    # 将模型输出的原始分数 logits 结合 temperatures（温度），通过 Softmax 函数转换为总和为 1 的概率分布
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)      # 在完整的概率分布上进行随机采样
    
    # 如果只指定了 Top-K，那么在概率分布上只考虑排名前 K 的词
    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    # 如果只指定了 Top-P，那么在概率分布上只考虑累积概率达到 P 的词
    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)
    
    # 当批次中存在既需要 Top-K 又需要 Top-P 截断的请求时，调用综合算子。
    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)

# 将批处理的请求参数转换为设备端（如 GPU）张量，并根据这些参数对模型的输出（Logits）执行具体的采样操作
@dataclass
class Sampler:
    device: torch.device        # 指定采样计算运行的设备
    vocab_size: int             # 模型的词表大小，用于处理未指定 top_k 时的默认边界值

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]    # 从批次中的每个请求（batch.reqs）提取采样配置
        if all(p.is_greedy for p in params):                # 检查批次中是否所有请求都采用贪心采样
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # 处理 top_k 参数，如果未指定则使用默认值（词表大小）
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]    # 将 top_p 的值严格限制在 [1e-6, 1.0] 的区间内
        temperatures = make_device_tensor(ts, torch.float32, self.device)       # 控制模型生成的随机性与确定性
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)        # Top-K 永远只看排名前 K 个词
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)      # Top-P 是动态的：候选词的数量会根据概率分布的形状自动缩放
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    # logits (torch.Tensor)：模型前向传播输出的原始打分矩阵
    # args (BatchSamplingArgs)：由 prepare 函数生成的包含批次采样参数（温度、Top-K、Top-P 张量）的数据类实例。
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):      # NVIDIA 提供的性能分析接口。它可以在使用 NVIDIA Nsight Systems 等工具进行 GPU 性能打点
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1) # 纯贪心采样，系统直接使用 torch.argmax 在词表维度（dim=-1）上找出分数最高的那个 Token 的索引
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
