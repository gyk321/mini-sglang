from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs          # 保存全局的服务器配置信息
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]     # 异步发送队列，用于向后端 Tokenizer 进程推送请求消息
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]      # 异步接收队列，用于从后端 Tokenizer 进程拉取响应消息
    uid_counter: int = 0        # 自增计数器，用于为每个新进入的用户请求分配全局唯一的 ID
    initialized: bool = False   # 标记后台监听协程 (listen) 是否已经启动
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)   # 响应暂存字典，将每个请求的 uid 映射到其待消费的生成结果列表
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)   # 事件映射字典，将每个请求的 uid 映射到其对应的 asyncio.Event 对象，用于协程间的同步与唤醒

    def new_user(self) -> int:
        uid = self.uid_counter      # 分配并返回一个新的 uid
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):                             # 后台常驻的无限循环协程
        while True:
            msg = await self.recv_tokenizer.get()       # 从异步 ZMQ Pull 队列（ZmqAsyncPullQueue）中拉取后端发回的消息
            for msg in _unwrap_msg(msg):                # 解包消息
                if msg.uid not in self.ack_map:         # 安全过滤
                    continue
                self.ack_map[msg.uid].append(msg)       # 将属于该 uid 的增量输出（UserReply）追加到该用户专属的缓冲列表中
                self.event_map[msg.uid].set()           # 事件唤醒，唤醒在等这个结果的协程

    def _create_listener_once(self):            # 惰性启动机制
        if not self.initialized:                # 确保后台接收协程已经就绪
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)      # 将消息放入异步 ZMQ Push 队列（ZmqAsyncPushQueue），非阻塞地发送给后端进程

    async def wait_for_ack(self, uid: int):     # 核心异步生成器
        event = self.event_map[uid]

        while True:
            await event.wait()                  # 挂起当前请求协程，等待listen()协程中event_map[msg.uid].set()唤醒（即后端返回了新的增量输出）
            event.clear()                       # 将状态重置回 False, 以便下一次等待新的增量输出时可以再次挂起

            pending = self.ack_map[uid]         # 取走刚刚 listen 放进去的数据
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack                       # 将增量输出（UserReply）逐个 yield 给调用者，供其处理或返回给前端（按下暂停键并递出物品）
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):  # 面向 /generate 接口和交互式 Shell 的流式生成器
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):  # OpenAI 兼容流式生成器
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):    # 客户端断连检测包装器
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():     # 检查客户端是否断开连接
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):   # 请求中止与清理方法
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))
        
    # 停止 send_tokenizer 和 recv_tokenizer 两个 ZMQ 队列，
    # 在 FastAPI 生命周期结束或 Shell 退出时调用
    def shutdown(self):     
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager        # 将一个生成器函数（包含 yield 的函数）转换为一个异步的上下文管理器
async def lifespan(_: FastAPI):
    yield
    # shutdown code here 在 FastAPI 服务器准备关闭时执行
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")      # 用来处理路径为 /generate 的 HTTP POST 请求
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()  # 获取全局的 FrontendManager 实例，负责管理与后端（Tokenizer/推理引擎）的 ZeroMQ 异步通信以及管理当前所有活跃的请求
    uid = state.new_user()      # 为当前这个生成请求分配一个全局唯一的 ID
    await state.send_one(
        TokenizeMsg(            # 将用户的请求打包成系统内部通信的“分词消息”
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(     # 采样参数
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",         # 打字机般“流式”推给客户端
    )

# FastAPI 中的一个“健康检查”或“基础路径探测”接口。它的主要作用是告诉客户端或监控系统：“API 的 v1 版本当前在线，并且运行正常”。
@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}

# 定义了这是一个处理 HTTP POST 请求的端点，路径完全兼容 OpenAI 的官方 API 标准
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()      # 负责管理与后端（Tokenizer/推理引擎）的 ZeroMQ 异步通信以及管理当前所有活跃的请求
    if req.messages:                # messages（Chat API 的标准做法，即包含了角色的对话历史）
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt         # 使用 req.prompt（普通的文本补全形式）

    # TODO: support more sampling parameters
    uid = state.new_user()          # 为这个请求分配一个唯一的 uid，并在状态机中为其注册用于接收结果的事件
    await state.send_one(
        TokenizeMsg(                # 将用户请求打包成一个 TokenizeMsg 消息
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )
    # 流式输出
    if req.stream:  
        return StreamingResponse(
            # stream_chat_completions 异步生成器，会不断监听后端传回来的数据块
            # stream_with_cancellation 在推送数据的同时，时刻监控 request.is_disconnected() 向后端发送中止信号 (AbortMsg) 以节省算力
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )
    # 非流式输出
    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        # 每当后端生成出一个片段（ack.incremental_output），就将其拼接到 full_content 字符串中
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")      # 用于列出当前可用的模型列表
async def available_models():
    state = get_global_state()
    # 启动服务器时通过命令行传入的模型路径或名称      
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]     # 将 Pydantic 的 Message 对象列表转换为标准的 Python 字典列表

    # TODO: support more sampling parameters
    uid = state.new_user()          # 为当前的 Shell 回复任务生成一个全局唯一的 ID
    # 将提取出的对话上下文以及采样参数封装成 TokenizeMsg 消息，通过 ZeroMQ 异步推送到后台，唤醒 GPU 开始推理。
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():                 # Shell 模式在生成过程中被强行中断或者生成结束时
        await state.abort_user(uid)     # 向后台发送一条 AbortMsg 消息，终止对应 uid 的计算，释放 GPU 资源

    return StreamingResponse(           # 返回流式响应对象
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)                     # 注册了两个内置命令。当用户在终端输入 / 时，按下 Tab 键可以自动补全
    session = PromptSession("$ ", completer=completer)      # 创建了一个交互式会话，终端的输入提示符被设置为 $。

    try:
        history: List[Tuple[str, str]] = []                 # 用来保存多轮对话历史的列表
        while True:
            cmd = (await session.prompt_async()).strip()    # 异步等待用户在终端输入内容
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            
            # 将历史对话转换为 Message 对象列表
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            # 将历史记录加上用户当前的新输入 cmd，一并打包进 OpenAICompletionRequest 请求体中
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )

            # 消费流式数据并实现“打字机”效果
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg

                # 手动解析 SSE 协议
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                # 保证每次打印不自动换行，flush=True 强制立刻刷新屏幕缓冲区。
                # 这样就能在终端里看到模型一个字一个字蹦出来的效果
                print(msg, end="", flush=True)
            print("", flush=True)
            # 生成结束后，把这一轮的用户输入和模型完整回复拼接存入 history，供下一轮对话使用
            history.append((cmd, cur_msg))
    # 异常捕获与彻底的资源清理
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()       # 停止与后台通信的 ZMQ 队列
        # then kill all the subprocesses
        import psutil
        # 清理僵尸进程：归查找并强制 kill() 当前进程产生的所有子进程，确保干干净净地退出
        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()

# 启动前端 API 服务器，通过 ZeroMQ (ZMQ) 将前端与后端的 Tokenizer 进程连接起来
def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host   # 主机地址
    port = config.server_port   # 端口号

    assert _GLOBAL_STATE is None, "Global state is already initialized" # 检查全局状态是否已经初始化，防止重复初始化
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(           # 接收来自后端的解码消息
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(           # 发送给后端的编码消息
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    # 实际上执行的就是 start_subprocess()
    start_backend()         # 启动后端工作进程（如张量并行调度器、分词器/解词器 Tokenizer/Detokenizer 等）

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
