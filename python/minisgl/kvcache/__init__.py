from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")

# 负责管理物理显存块
def create_kvcache_pool(
    model_config: ModelConfig,      # 模型的结构超参数
    num_pages: int,                 # 显存池预先分配的物理页（Page/Block）总数量，决定了缓存池的最大容量
    page_size: int,                 # 每个物理页能够存储的 Token 数量（即 PagedAttention 中的 Block Size）
    dtype: torch.dtype,             # 缓存张量的数据类型
    device: torch.device,           # 显存池所在的计算设备
) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,     # Key 和 Value 的注意力头数量（支持 MHA、GQA 或 MQA 架构）
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,         # Transformer 的总层数（每一层都需要独立的 KV 缓存空间）
        head_dim=model_config.head_dim,             # 每个注意力头的特征维度大小
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")          # 将自身注册到系统支持的缓存管理器列表中
def create_naive_cache(device: torch.device):       # 负责管理逻辑页表与前缀复用
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)

@SUPPORTED_CACHE_MANAGER.register("radix")          # 将自身绑定到键值 "radix"
def create_radix_cache(device: torch.device):       # 创建一个基于基数树（Radix Tree）的前缀缓存管理器
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)

# 负责管理逻辑页表与前缀复用
def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
