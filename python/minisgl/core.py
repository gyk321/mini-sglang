from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:           # 封装并传递大语言模型在生成（Decode）阶段的采样策略和超参数
    temperature: float = 0.0    # 温度参数，控制生成的随机性
    top_k: int = -1             # 在采样下一个 Token 时，仅从概率最高的 k 个候选 Token 中进行选择
    top_p: float = 1.0          # 核采样（Nucleus Sampling）参数,将候选词按概率从高到低排列并累加，模型只从累计概率达到 p 的前几个词中进行随机采样，从而过滤掉低概率的离谱词汇
    ignore_eos: bool = False    # 是否忽略模型输出的结束符
    max_tokens: int = 1024      # 单次请求允许生成的最大 Token 数量，用于控制输出长度上限

    @property
    def is_greedy(self) -> bool:    # 判断当前的采样参数组合是否等效于“贪婪解码”
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0

# 推理引擎中追踪单条语句生成生命周期的核心数据结构
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # 当前请求的输入 Token ID 序列
    table_idx: int          # 当前请求在页表中的索引
    cached_len: int         # 前半段的长度
    output_len: int         # 后半段的长度，也就是这一轮要生成的 Token 数量
    uid: int                # 当前请求的唯一标识符
    sampling_params: SamplingParams     # 当前请求的采样参数，包含温度、top-k、top-p 等信息
    cache_handle: BaseCacheHandle       # 当前请求在前缀缓存中的句柄，用于访问和管理该请求的缓存数据

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu                # 强制确保输入的张量位于 CPU 上
        self.device_len = len(self.input_ids)       # 当前这轮要处理到的总长度
        self.max_device_len = len(self.input_ids) + self.output_len             # 请求最终能达到的最大长度（当前输入长度 + 期望生成的输出长度）
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len    # 已缓存的长度不能超过当前拥有的长度，且当前长度不能超过允许的最大总长度

    @property
    def remain_len(self) -> int:        # 剩余需要生成的 Token 数量
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:                    # 当前需要进行 Prefill（预填充）或计算注意力的 Token 数量，通常代表还没被计算并写入 KV Cache 的新 Token 数
        return self.device_len - self.cached_len
    
    # 在解码（Decode）阶段，当模型生成了一个新的 Token 后，调用该方法将当前请求的状态更新为已处理一个 Token，
    # 即将 device_len 增加 1，同时更新 cached_len 为新的 device_len
    def complete_one(self) -> None:         
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:    # 将模型新生成的一个 Token 拼接到当前的 input_ids 序列的末尾
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:                   # 判断该请求是否还没有达到最大生成长度，是否还可以继续进行 Decode（解码）阶段
        return self.remain_len > 0

    def __repr__(self) -> str:          # 返回当前请求的字符串表示形式，包含请求的页表索引、已缓存长度、当前处理长度以及最大处理长度等信息
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )

# 将多个单独的请求 (Req) 打包组合在一起，以便利用 GPU 并行处理，从而提高大模型推理的吞吐量
@dataclass
class Batch:
    reqs: List[Req]                                 # 该批次包含的请求列表，里面的每一个元素都是之前解释过的 Req 对象
    phase: Literal["prefill", "decode"]             # 标识当前这个批次所处的计算阶段
    # these fields should be set by scheduler       
    input_ids: torch.Tensor = field(init=False)     # 当前批次所有请求组合后的输入 Token 序列张量
    positions: torch.Tensor = field(init=False)     # 与 input_ids 对应的位置编码张量（用于旋转位置编码 RoPE 等计算）
    out_loc: torch.Tensor = field(init=False)       # 输出位置信息的张量。通常用于在底层连续内存中追踪当前批次输出结果需要写入或读取的具体位置
    padded_reqs: List[Req] = field(init=False)      # 经过填充（Padding）后的请求列表
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)     # 由注意力计算后端动态填充。它包含了底层注意力算子（如 PagedAttention）运行所需的元数据

    @property
    def is_prefill(self) -> bool:       # 判断当前批次是否处于预填充
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:        # 判断当前批次是否处于解码（Decode）阶段
        return self.phase == "decode"

    @property
    def size(self) -> int:              # 返回当前批次中实际包含的请求数量
        return len(self.reqs)

    @property
    def padded_size(self) -> int:       # 返回经过底层填充要求对齐后的请求总数量
        return len(self.padded_reqs)

# 存储底层硬件计算后端、管理显存池（KV Cache），并在模型的前向传播（Forward Pass）中安全地追踪和调度当前正在处理的批次
@dataclass
class Context:
    page_size: int      # KV Cache 的页大小
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)            # 全局页表张量，用于记录逻辑 Token 到物理显存块的映射关系（PagedAttention 的核心数据结构）
    attn_backend: BaseAttnBackend = field(init=False)       # 注意力计算后端（例如基于 FlashInfer、Triton 等底层算子封装的类），负责执行高效的 Attention 计算
    moe_backend: BaseMoeBackend = field(init=False)         # 混合专家（Mixture of Experts）计算后端，用于处理 MoE 架构模型的路由和专家计算
    kv_cache: BaseKVCachePool = field(init=False)           # KV Cache 显存池，负责 GPU 上 KV 张量内存的分配和释放
    _batch: Batch | None = field(default=None, init=False)  # 用于临时存储当前正在被模型处理的 Batch 对象

    @property
    def batch(self) -> Batch:       # 安全获取当前正在处理的批次
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):  # 在模型的前向传播期间，安全地管理和追踪当前正在处理的批次的状态
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None  # 保存全局唯一的 Context 实例


def set_global_ctx(ctx: Context):   # 设置全局上下文，且只能设置一次（防止被意外覆盖）
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:    # 获取全局上下文
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
