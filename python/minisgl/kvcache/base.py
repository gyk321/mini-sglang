from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...      # 根据给定的索引 (index) 检索并返回对应的 Key 缓存张量

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...      # 根据给定的索引 (index) 检索并返回对应的 Value 缓存张量

    @abstractmethod
    def store_kv(                                           # 将新的 Key (k) 和 Value (v) 存储到缓存中
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...                   # 返回该 KV Cache 所在的计算设备

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...                     # 返回缓存张量的数据类型

    @property
    @abstractmethod
    def num_layers(self) -> int: ...                        # 返回该缓存池维护的 Transformer 层数

# 充当缓存中特定前缀数据的“句柄”
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...      # 提取这段缓存真正在物理存储（比如 GPU 显存）中对应的索引位置

# 报告当前缓存的容量与占用状态
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size    # 返回缓存的已用总大小（即可驱逐大小与受保护大小之和）

# 插入新前缀
class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix

# 查找匹配的前缀
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁特定的缓存句柄 (BaseCacheHandle)。
        为了防止正在被使用的数据被清理，在调用匹配结果前必须锁定句柄。
        该操作仅更新大小状态，不修改实际缓存内容。
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        在缓存中查找与 input_ids 匹配的前缀，并返回匹配结果
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        将新的前缀序列以及它们对应的物理存储索引 (indices) 记录到缓存管理器中，并返回插入状态 (InsertResult)。
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        当缓存空间不足时，驱逐指定大小的数据以释放空间，并返回被驱逐掉的物理索引。
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """完全重置缓存管理器和底层缓存。Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """
        返回当前缓存的大小分布信息，
        包括可被驱逐的大小 (evictable_size) 和受保护/被锁定的状态大小 (protected_size)。
        Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """用于内部校验，检查缓存数据结构是否损坏。Check the integrity of the cache. Raise an error if the cache is corrupted."""
