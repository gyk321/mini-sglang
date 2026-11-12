from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)

# 当一个用户的长文本请求在 PrefillAdder 中分配资源时，
# 如果当前批次的 token_budget（算力预算）不足以一次性处理完它所有的 Token，
# 系统就会将其标记并实例化为 ChunkedReq，而不是普通的 Req
class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:    # 在解码（Decode）阶段，将模型新生成的一个 Token 附加到该请求的主机端记录中
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:       # 判断该请求是否可以进入解码（Decode）阶段
        return False  # avoid being added to decode manager

# 用于在调度过程中尝试将待调度的请求（PendingReq）添加到当前批次中，并根据可用的 Token 预算和显存资源来决定是否可以添加请求
@dataclass
class PrefillAdder:        
    token_budget: int       # 当前批次（Batch）还剩余的可处理 Token 数量
    reserved_size: int      # 必须预留的 Token 容量大小
    cache_manager: CacheManager     # 管理前缀缓存
    table_manager: TableManager     # 管理底层的显存页表映射

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:                  # 如果当前页表已满，则无法为新的请求分配显存资源，直接返回 None
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle      # 尝试在前缀缓存中匹配当前请求的输入数据，如果匹配成功，则返回一个 BaseCacheHandle 实例，否则返回 None
        cached_len = handle.cached_len                              # 获取该请求在前缀缓存中已经匹配到的 Token 长度
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len                     # Prompt 中没有被缓存命中的部分长度（需要新计算预填充的长度）
        estimated_len = extend_len + req.output_len                 # 未命中的 Prompt 长度 + 预估要生成的输出长度

        # 第一次显存安全检查.如果当前请求的预估总长度（未命中部分 + 输出长度）加上必须预留的显存容量超过了 CacheManager 的可用显存容量，
        # 则无法为该请求分配显存资源，直接返回 None
        if estimated_len + self.reserved_size > self.cache_manager.available_size:  
            return None
        self.cache_manager.lock(handle)     # 将匹配到的缓存节点锁定（通常是增加引用计数），防止它们在接下来被 LRU 等缓存淘汰策略清理掉
        if estimated_len + self.reserved_size > self.cache_manager.available_size:      # 第二次显存检查
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()       # 尝试为当前请求分配一个新的页表索引（table_idx），用于在底层显存中存储该请求的 Token 数据
        if cached_len > 0:  # NOTE: set the cached part 命中了前缀缓存
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]              # 获取当前请求在底层显存中对应的 Token 存储区域（device_ids），用于存放已经命中的前缀缓存部分
            page_entry = self.table_manager.page_table[table_idx][:cached_len]              # 获取当前请求在页表中对应的物理页索引（page_entry），用于记录已经命中的前缀缓存部分在底层显存中的位置
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)    # 将当前请求的输入数据中已经命中的前缀缓存部分（req.input_ids[:cached_len]）复制到底层显存的对应位置（device_ids），以便后续计算使用
            page_entry.copy_(handle.get_matched_indices())                                  # 将当前请求的页表索引（page_entry）更新为前缀缓存中已经命中的物理页索引（handle.get_matched_indices()），确保页表正确映射到底层显存

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,        # 等待处理的原始请求
        cache_handle: BaseCacheHandle,  # 已经锁定好的缓存句柄（用于前缀复用）
        table_idx: int,                 # 已经在显存页表中为该请求分配好的行索引
        cached_len: int,                # 命中的前缀缓存长度（这部分 Token 不需要再计算）
    ) -> Req:       # 返回一个可以直接送入底层模型执行的 Req（普通请求）或 ChunkedReq（分块请求）
        remain_len = pending_req.input_len - cached_len     # 当前请求真正需要进行预填充计算的长度（总长度减去已经有缓存的长度）
        chunk_size = min(self.token_budget, remain_len)     # 计算当前批次中可以处理的最大 Token 数量（chunk_size）
        is_chunked = chunk_size < remain_len                # 判断当前请求是否需要被分块处理
        CLS = ChunkedReq if is_chunked else Req             # 根据是否被截断，决定最终实例化的类是 ChunkedReq 还是 Req
        self.token_budget -= chunk_size                     # 更新当前批次的剩余 Token 预算，减去本次处理的 chunk_size
        self.reserved_size += remain_len + pending_req.output_len       # 更新必须预留的显存容量，增加当前请求的未命中长度和输出长度，以确保后续调度不会超过可用显存
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size) # 确定本次需要计算的 Token 切片范围
        device_ids = self.table_manager.token_pool[table_idx, _slice]   # 定位到 GPU 上的 token_pool 中对应的物理位置
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True) # 将当前请求的输入数据中未命中的部分（pending_req.input_ids[_slice]）复制到底层显存的对应位置（device_ids），以便后续计算使用
        return CLS(     # 将所有上下文信息（本次要处理的 input_ids、页表索引、缓存信息、采样参数等）打包成 Req 或 ChunkedReq 实例并返回
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:   # 调度器（PrefillManager）会循环调用它来尝试装载请求
        if self.token_budget <= 0:  # 算力耗尽
            return None
        # 尝试处理尚未完成的分块请求（Chunked Request）
        if chunked_req := pending_req.chunked_req:      # 如果当前请求已经被分块处理过（即 chunked_req 不为 None）
            return self._add_one_req(                   # 继续处理剩下的部分
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )
        # 尝试处理全新的请求（需要分配显存）
        if resource := self._try_allocate_one(pending_req): # 尝试为当前请求分配显存资源，如果成功，则返回一个包含缓存句柄和页表索引的元组，否则返回 None
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )
        # 分配失败
        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager     # 用于管理 KV Cache 的 CacheManager 实例，负责处理缓存的分配、释放和匹配等操作
    table_manager: TableManager     # 用于管理请求的页表的 TableManager 实例，负责维护每个请求在物理显存中的页表索引和分配状态
    decode_manager: DecodeManager   # 用于管理解码任务的 DecodeManager 实例，负责处理自回归生成过程中的解码操作和任务调度
    pending_list: List[PendingReq] = field(default_factory=list)    # 用于存储当前待调度的请求列表，每个请求包含输入数据、采样参数以及可选的分块请求信息

    def add_one_req(self, req: UserMsg) -> None:    # 将一个新的请求（UserMsg）添加到待调度的请求列表中，创建一个 PendingReq 实例并将其加入 pending_list，以便后续进行调度和处理
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))
    # 尝试从待调度的请求列表中调度出一个新的批次（Batch），根据给定的 prefill_budget（预填充预算）来决定可以处理的请求数量和大小，
    # 如果成功调度出请求，则返回一个 Batch 实例，否则返回 None
    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:     
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        # 用于尝试将待调度的请求添加到当前批次中，并根据预填充预算和解码管理器的状态来决定是否可以添加请求
        adder = PrefillAdder(           
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []                        # 用于存储当前批次中成功调度的请求列表
        chunked_list: List[PendingReq] = []         # 用于存储当前批次中被分块处理的请求列表
        for pending_req in self.pending_list:       # 遍历待调度的请求列表，尝试将每个请求添加到当前批次中
            if req := adder.try_add_one(pending_req):   # 如果成功添加请求，则将其加入 reqs 列表，并根据请求类型（是否为分块请求）将其加入 chunked_list 列表
                pending_req.chunked_req = None      # 首先清空该请求之前的 chunked_req 状态
                if isinstance(req, ChunkedReq):     # 检查返回的 req 是否是 ChunkedReq（分块请求）
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)    # 把这个原始请求加入到 chunked_list
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]   # 更新等待队列 (处理未完成的任务)
        return Batch(reqs=reqs, phase="prefill")
    # 根据请求的唯一标识符（uid）从待调度的请求列表中移除对应的请求，并返回该请求的分块请求信息（ChunkedReq），
    # 如果未找到对应的请求，则返回 None
    def abort_req(self, uid: int) -> Req | None:    
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
