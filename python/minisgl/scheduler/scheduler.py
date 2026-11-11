from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"

# 推理框架的核心控制器：负责统筹整个 LLM 推理的生命周期，包括接收用户请求、管理显存（KV Cache 与 Page Table）、
# 调度 Prefill（预填充）和 Decode（解码）批次、驱动底层引擎执行前向计算，
# 以及通过重叠调度（Overlap Scheduling）隐藏 CPU 开销以最大化 GPU 利用率
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)        # 负责底层模型的前向推理、采样、CUDA Graph 填充以及 Attention 元数据准备

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # 引擎计算流
        torch.cuda.set_stream(self.stream)  # 调度器流，被设为默认流

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)   # 管理请求在全局页表中的槽位以及全局 Token 池
        self.cache_manager = CacheManager(                          # 管理分页 KV Cache（Paged Attention 的物理页分配、前缀缓存复用与释放）
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)       # 维护处于生成阶段（Decode）的请求队列并负责构建 Decode 批次
        self.prefill_manager = PrefillManager(                      # 维护新到达或分块预填充（Chunked Prefill）的请求队列
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
    # 重叠调度（Overlap Scheduling）主循环函数。它的核心思想是通过双 CUDA 流构建流水线，
    # 将当前批次的 GPU 前向计算与上一批次结果的 CPU 后处理（以及下一批次的 CPU 调度准备）并行重叠，
    # 从而隐藏 CPU 开销，让 GPU 保持满载运行。
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # 只有当系统完全空闲（既无上一轮遗留结果，也无任何待跑请求）时，blocking 才为 True
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed 当前有上一批待处理的数据（last_data）
            or self.prefill_manager.runnable        # 有待执行的 Prefill 请求
            or self.decode_manager.runnable         # 有待执行的 Decode 请求
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)          # 趁着 GPU 正在后台繁忙计算第 N 批（ongoing_data），CPU 立即向下执行去处理第 N-1 批的结果
        return ongoing_data

    def normal_loop(self) -> None:
                    # 是否有待预填充的新请求或分块请求      是否有正在生成 Token 的活跃请求
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):     # 遍历从前端/通信队列收到的所有消息,自适应阻塞接受消息
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()         # 调度并准备下一批次
        ongoing_data = None                                 # 初始化当前执行状态，用于保存当前批次的输入信息与 GPU 推理输出句柄。
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)               # 同步并处理当前批次结果

    @torch.inference_mode()     # 完全禁用自动求导（Autograd）引擎，关闭了张量版本追踪（Version Tracking）和视图追踪
    def run_forever(self) -> NoReturn:      # 调度器的入口函数
        if ENV.DISABLE_OVERLAP_SCHEDULING:  # 禁用重叠调度,系统退化为传统的单流串行执行模式
            with self.engine_stream_ctx:    # 将全局当前 CUDA 流切换回引擎的计算流
                self.engine.stream.wait_stream(self.stream)     # 确保此前在 self.stream(调度流) 上排队的所有初始化或异步拷贝操作全部完成后，计算流再开始工作
                while True:
                    # 串行循环：此时，每一轮的数据准备（_prepare_batch）、模型前向计算（_forward）
                        # 和结果后处理（_process_last_data）全部在同一个流（self.engine.stream）上按顺序串行执行，
                        # 清晰易调试但无法掩盖 CPU 延迟
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream   # 确认当前默认 CUDA 流确实是调度器专用的调度流
            data = None
            while True:
                data = self.overlap_loop(data)      # 重叠调度循环:在每一轮迭代中，它实现了“第 N 批的 GPU 计算”与“第 N-1 批的 CPU 后处理”相重叠
    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        # 解包与同步:next_tokens_cpu（位于 CPU 锁页内存的采样结果张量）和 copy_done（CUDA Event）
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()             # 阻塞当前 CPU 线程
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():     # 延迟释放上下文
            for i, req in enumerate(batch.reqs):        # 遍历请求与更新主机端 Token
                if isinstance(req, ChunkedReq):         # 跳过未完成的分块请求
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))    # 取出当前请求生成的 next_token 并将其追加到请求在 CPU 侧维护的完整序列中
                next_token = int(next_token.item())     # 转为 Python int 类型
                finished = not req.can_decode           # 长度检查,append_host 更新长度后，如果请求已达到设定的最大生成长度（max_tokens）或模型最大序列长度，req.can_decode 会变为 False
                if not req.sampling_params.ignore_eos:  # EOS 终止符检查
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))  # 打包消息,加入 reply 列表

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                # （重叠调度）模式下，CPU 总是先调度并启动第 N+1 步的 GPU 计算，然后才调用 _process_last_data 去检查第 N 步的计算结果。这会导致一个时间差问题
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req) # 将其从 Decode 活跃队列中移除
                    self._free_req_resources(req)       # 归还该请求占用的 table_idx 槽位，并将完整序列的 KV Cache 释放/转入缓存树
                    new_finished_reqs.add(req)          # 将该请求加入本轮新完成集合
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs          # 用本轮新完成的请求集合覆盖上一轮的记录
        self.send_result(reply)                         # 通过进程间通信将 reply 列表中的所有 DetokenizeMsg 批量发送给下游的 Detokenizer 进程进行文本解码与流式输出。

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):            # 处理批量消息,通过递归的方式调用自身
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):                  # 终止整个调度器
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):                  # 一个新到达的用户文本生成请求
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len    # 获取当前输入的 Token 长度,引擎支持的最大序列长度
            max_output_len = max_seq_len - input_len    # 计算出该请求允许生成的最大输出长度
            if max_output_len <= 0:                     # 输入的 Prompt 长度已经达到或超过了引擎的最大限制
                return logger.warning_rank0(            # 记录一条警告日志并直接丢弃该请求
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:     # 如果用户请求中指定的期望生成长度大于计算出的可用剩余长度
                msg.sampling_params.max_tokens = max_output_len     # 强制将其缩小至 max_output_len 以防止显存溢出
                logger.warning_rank0(                               # 打印警告日志
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)                   # 将该消息作为新请求加入到 prefill_manager（预填充阶段管理器）的等待队列中，等待后续调度执行
        elif isinstance(msg, AbortBackendMsg):                      # 客户端主动取消某个请求，或者请求因异常需要被终止
            logger.debug_rank0("Aborting request %d", msg.uid)      # 打印调试日志
            req_to_free = self.prefill_manager.abort_req(msg.uid)   # 根据消息提供的唯一标识符 msg.uid，首先尝试从 prefill_manager 中将其移除
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid) # 如果在 prefill_manager 中未找到该请求，则继续尝试从 decode_manager 中将其移除
            if req_to_free is not None:                             # 如果找到了该请求对象，则调用 _free_req_resources 方法释放其占用的资源（如页表槽位和 KV Cache）
                self._free_req_resources(req_to_free)
        else:                                                       # 如果收到的消息类型不在上述已知类型中，则记录一条错误日志并抛出 NotImplementedError 异常，提示该消息类型尚未被处理
            logger.error(f"Unknown message type: {type(msg)}")      # 记录错误日志，显示未知消息类型
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)               # 补齐批次大小
        self.cache_manager.allocate_paged(batch.reqs)           # 分配物理显存页
        batch.positions = _make_positions(batch, self.device)   # 位置编码
        input_mapping = _make_input_tuple(batch, self.device)   # 输入读取映射
        write_mapping = _make_write_tuple(batch, self.device)   # 输出写回映射
        batch.out_loc = self.engine.page_table[input_mapping]   # 对应的物理缓存位置
        self.engine.attn_backend.prepare_metadata(batch)        # 准备算子与采样元数据
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)   # 第一优先级（Prefill 阶段）
            or self.decode_manager.schedule_next_batch()                    # 第二优先级（Decode 阶段）:将当前所有处于逐 Token 生成阶段的活跃请求打包成一个 Decode
        )
        # 有任务可跑 (batch 存在)：将逻辑上的请求集合转化为 GPU 推理所需的底层物理数据结构，最终返回 ForwardInput 对象。
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input   # 解包输入参数
        batch.input_ids = self.token_pool[input_mapping]                    # 从全局 Token 池读取本步输入
        forward_output = self.engine.forward_batch(batch, sample_args)      # 执行真正的 GPU 计算（包括 Embedding、Transformer 层前向传播、写入 KV Cache、计算 Logits 以及 Token 采样）
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu    # 新生成的 Token 原地写回显存池
        self.decode_manager.filter_reqs(forward_input.batch.reqs)           # 更新解码队列
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
