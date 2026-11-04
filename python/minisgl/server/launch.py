from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs

# 每个调度器子进程（Scheduler Process）实际执行的核心工作负载
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():        # 全局禁用梯度计算
        scheduler = Scheduler(args)     # 实例化了核心的 Scheduler 对象：负责管理请求队列、动态批处理、显存分页以及执行模型的计算
        scheduler.sync_all_ranks()      # 分布式屏障:确保所有参与并行的 GPU 都已经成功加载了模型权重并准备就绪

        if args.tp_info.is_primary():   # is_primary()（通常判断是否为 Rank 0 主节点）限制了只有 1 个进程会发送 ack 消息
            ack_queue.put("Scheduler is ready")     # 向主进程的 ack_queue 队列发送一条确认消息

        if args.silent_output:          # 静音信息级日志
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()     # 函数在此进入死循环
        except KeyboardInterrupt:       # Ctrl+C（即 KeyboardInterrupt 异常）
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")      # 打印一条友好的退出信息
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)    # 强制使用 "spawn" 模式启动多进程

        world_size = server_args.tp_info.size       # 获取分布式推理时的张量并行（Tensor Parallelism，简称 TP）的规模
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()       # 多进程队列，用于接收各个子进程准备就绪的确认信号

        for i in range(world_size):                 # 根据张量并行（Tensor Parallelism）的配置大小，循环启动多个调度器进程
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # 启动反词元化器 
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        # 启动词元化器
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()
        # 等待 num_tokenizers + 2 个信号（包含：1 个来自主节点调度器的信号 + 1 个反词元化器信号 + num_tokenizers 个词元化器信号）
        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
