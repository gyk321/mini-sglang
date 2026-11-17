from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")

# 同步消息发送队列
class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,      # ZeroMQ 的通信地址
        create: bool,   # 决定当前端点的连接模式,True作为服务端绑定并创建端口；False作为客户端连接到已有端口
        encoder: Callable[[T], Dict],   # 序列化转换函数
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH) # PUSH 套接字遵循单向推送模式
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):      # 发送数据
        event = msgpack.packb(self.encoder(obj), use_bin_type=True) # packb将字典编码为紧凑的二进制字节流
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
# 异步消息接收队列封装类
# Generic[T] 使用者在实例化时可以指定队列中传输的具体数据类型
class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,          # ZeroMQ 的通信地址
        create: bool,       # 决定当前节点在连接中的角色
        decoder: Callable[[Dict], T],           # 反序列化回调函数
    ):
        # 负载均衡地从上游拉取任务/消息，构成单向的流水线（Pipeline）消息分发模型
        self.context = zmq.asyncio.Context()    # 创建异步上下文
        self.socket = self.context.socket(zmq.PULL) # 实例化一个 zmq.PULL 类型的 Socket
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:   # 异步接收字节流
        event = await self.socket.recv()    # 挂起当前协程，直到 Socket 接收到一条完整的二进制消息
        # MessagePack 解包：将接收到的二进制数据（bytes）反序列化为标准的 Python 字典
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()     # 关闭 ZeroMQ Socket
        self.context.term()     # 终止异步上下文


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
