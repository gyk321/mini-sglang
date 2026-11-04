from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]

# 作为独立的工作进程运行，充当前端（Frontend）与后端推理引擎（Backend）之间的中间件，
# 专门负责异步处理文本的编码（Tokenization）、解码（Detokenization）以及请求中断信号的转发
@torch.inference_mode()     # 禁用 PyTorch 的梯度计算与视图追踪
def tokenize_worker(
    *,
    tokenizer_path: str,        # 指定分词器模型路径
    addr: str,                  # 定义当前 Worker 监听的 ZeroMQ 地址
    create: bool,
    backend_addr: str,          # 后端推理引擎地址
    frontend_addr: str,         # 前端服务地址
    local_bs: int,              # 控制单次从队列拉取并聚合处理的最大消息数
    tokenizer_id: int = -1,     # 用于日志标识
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,     # 多进程队列，用于在分词器加载和初始化完成后向主进程发送就绪信号
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)     # 创建两个推送队列
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)        # 拉取队列,用于接收传入的 BatchTokenizerMsg
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)      # 加载分词器实例
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager       
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)       # 负责文本转 Token Tensor
    detokenize_manager = DetokenizeManager(tokenizer)   # 负责增量 Token 转文本

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")   # 通知主控进程该 Worker 已准备好接收请求

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())                      # 阻塞等待第一条消息
            while len(pending_msg) < local_bs and not recv_listener.empty():    # 在不阻塞的前提下尽可能多地拉取当前积压的消息，凑成一个批次以提高吞吐量
                # 使用 Python 列表的 extend 方法，将 _unwrap_msg 返回列表中的所有元素逐个追加到 pending_msg 末尾
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]   # 来自后端的生成结果，需解码发往前端
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]       # 来自前端的新用户请求，需编码发往后端
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]             # 取消生成的指令，需转发给后端
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)

            # 解码分支（发往 Frontend）
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                # 单条消息优化:如果批次中只有一条消息，则直接剥离外层的 Batch*Msg 包装发送单个对象，减少序列化开销
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            # 编码分支（发往 Backend）
            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            # 中断分支（发往 Backend）
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
