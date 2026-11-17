from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:     # 离线模式
            self.receive_msg = self.offline_receive_msg     # 从前端/分词器接收请求消息。支持阻塞和非阻塞
            self.send_result = self.offline_send_result     # 将推理生成的 Token 结果发回给去分词器
            return  # early exit

        if tp_info.is_primary():    # 检查当前进程是否是主进程
            # Final：类型注解，表明该属性在初始化后不应被重新赋值或覆盖
            self._recv_from_tokenizer: Final = ZmqPullQueue(    # 单向拉取 Tokenizer 发送过来的推理请求
                config.zmq_backend_addr,                        # 通信地址，指定后端监听的位置
                create=True,        # 由当前 Scheduler 进程负责创建（bind）这个通信端点，等待 Tokenizer 进程来连接
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(    # 将调度器生成的结果单向推送给 Detokenizer 进程
                config.zmq_detokenizer_addr,                    # 指定目标 Detokenizer 进程的通信地址
                create=config.backend_create_detokenizer_link,  # 动态决定是由当前后端来创建连接，还是由 Detokenizer 创建而当前后端主动去连接
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:            # 多卡张量并行模式
            if tp_info.is_primary():    # 主节点
                recv = self._recv_msg_multi_rank0
                # 额外创建一个 ZmqPubQueue，负责将收到的外部请求广播给其他计算卡
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                # 不直接与外部的分词器建立连接，而是初始化一个 ZmqSubQueue（订阅者队列）来监听 Rank 0 的广播
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())    # 阻塞挂起当前线程，直到 ZMQ 队列中至少到达 1 条新请求消息为止
        while not self._recv_from_tokenizer.empty():                # 检查队列是否非空
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs         # 把当前积压在 ZMQ 队列里的所有剩余消息一次性全部取出

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []         # 用于存放本轮解码后需要处理的所有消息
        if blocking:
            self.run_when_idle()                        # 执行一些后台空闲维护任务
            raw = self._recv_from_tokenizer.get_raw()   # 从分词器拉取第一条原始字节流
            self._send_into_ranks.put_raw(raw)          # 通过 ZMQ Pub 队列（_send_into_ranks）立即将这条原始消息广播出去
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))  # 将原始字节解码为 BaseBackendMsg 对象，加入待处理列表

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            # 将当前队列中所有未读的原始字节消息全部取出，实现批量拉取（Batching），提高吞吐量
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks 同步消息数量（关键的分布式对齐）
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:            # 遍历刚才暂存的 pending_raw_msgs
            self._send_into_ranks.put_raw(raw)  # 逐条通过 ZMQ Pub 队列广播给其他 Rank
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))  # 在本地完成解码并追加到 pending_msgs
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []     # 用于收集本轮需要处理的消息对象
        if blocking:
            self.run_when_idle()                    # 执行本地的空闲后台任务
            # 从订阅 Rank 0 的 ZMQ Sub 队列中读取第一条消息，并将其自动反序列化（Decode）为 BaseBackendMsg 对象
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        dst_tensor = torch.tensor(-1)       # 创建一个占位符 Tensor
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()  # 阻塞等待 Rank 0 将本轮排空队列后得到的剩余消息总数广播过来
        dst_length = int(dst_tensor.item()) # 将接收到的 Tensor 转换为 Python 整数

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())    # 从 _recv_from_rank0 中依次取出并解码所有剩余消息
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        num_reply = len(reply)      # 获取本轮调度步骤（Step）中生成的待回复消息数量
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:          # 只有 1 条消息
            # 将单个 DetokenizeMsg 列表封装成一个 BatchTokenizerMsg 批量消息对象，再一次性投入 ZMQ 队列
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:         # 有多条消息
            # 将多条 DetokenizeMsg 列表封装成一个 BatchTokenizerMsg 批量消息对象，再一次性投入 ZMQ 队列
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
