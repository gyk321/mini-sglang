from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str             # 模型的路径或 Hugging Face 上的模型名称
    tp_info: DistributedInfo    # 张量并行的分布式信息
    dtype: torch.dtype          # 模型加载和计算时使用的数据类型（如 torch.float16, torch.bfloat16 等）
    max_running_req: int = 256  # 引擎同时处理的最大运行请求数
    attention_backend: str = "auto"     # 指定注意力计算的后端实现方式，可选值包括 "auto"、"flash"、"triton" 等
    moe_backend: str = "auto"           # 指定 MoE（Mixture of Experts）计算的后端实现方式，可选值包括 "auto"、"torch"、"triton" 等
    cuda_graph_bs: List[int] | None = None  # 指定 CUDA 图（CUDA Graph）优化的批次大小列表，如果为 None，则不启用 CUDA 图优化
    cuda_graph_max_bs: int | None = None    # 指定 CUDA 图优化的最大批次大小，如果为 None，则不限制最大批次大小
    page_size: int = 1                      # KV Cache 分页的大小，单位为 Token 数量，决定了每个物理页中可以存储的 Token 数量
    memory_ratio: float = 0.9               # 指定 KV Cache 分配的显存比例，范围为 0.0 到 1.0，表示可用于 KV Cache 的显存占总显存的比例
    distributed_timeout: float = 60.0       # 指定分布式通信的超时时间，单位为秒，用于控制在分布式环境下的通信操作的最长等待时间
    use_dummy_weight: bool = False          # 指定是否使用虚拟权重（Dummy Weight）进行模型加载和计算，主要用于调试和测试目的
    use_pynccl: bool = True                 # 指定是否使用 PyNCCL 库进行分布式通信，PyNCCL 是 NVIDIA 提供的高性能通信库，用于在多 GPU 环境下进行高效的数据传输
    max_seq_len_override: int | None = None # 如果不为 None，则会覆盖模型配置中的最大序列长度（max_seq_len），用于在特定场景下限制输入序列的最大长度
    num_page_override: int | None = None  # if not None, will override the number of pages 强制覆盖系统自动计算出的总物理页数量

    @cached_property
    def hf_config(self):        # 加载并缓存 Hugging Face 的模型配置文件
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:  # 加载并缓存模型配置
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:           # 返回模型的最大序列长度，如果 max_seq_len_override 不为 None，则使用该值覆盖模型配置中的最大序列长度
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:       # 返回模型的最大前向长度（max_forward_len），默认情况下等于模型的最大序列长度（max_seq_len），但可以在子类中进行覆盖以实现不同的行为
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:      # 返回分布式通信的地址，默认情况下为 "tcp://127.0.0.1:2333"
        return "tcp://127.0.0.1:2333"
