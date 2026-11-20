from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int                                          # KV Cache 分页管理
    running_reqs: Set[Req] = field(default_factory=set)     # 当前处于解码运行状态的请求（Req 对象）集合

    def filter_reqs(self, reqs: Iterable[Req]) -> None:     # 更新并过滤活跃的解码请求池
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:                 # 安全移除指定的请求
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:            # 根据唯一标识符（uid）中止并移除特定请求
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:                       # 计算当前所有运行中解码请求在未来可能占用的最大 Token 容量
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:          # 构建下一个用于模型前向传播的 Decode 批次
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:                             # 判断当前是否存在可调度的解码请求
        return len(self.running_reqs) > 0
