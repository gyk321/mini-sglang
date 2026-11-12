from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192       # 指定调度器在处理请求时允许的最大扩展 Token 数量，超过该数量的请求将被拒绝或截断
    cache_type: str = "radix"           # 指定调度器使用的缓存类型，可选值包括 "radix"、"lru" 等，决定了缓存的替换策略和数据结构
    offline_mode: bool = False          # 指定调度器是否以离线模式运行，离线模式下调度器不会与外部系统进行实时通信，而是依赖于预先加载的数据和配置

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)    # 用于生成唯一的调度器实例标识符，通常基于进程 ID，以避免在多实例运行时的资源冲突

    @property
    def zmq_backend_addr(self) -> str:  # 返回调度器的 ZeroMQ 后端通信地址
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:  # 返回调度器的 ZeroMQ 解码器通信地址
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:      # 返回调度器的 ZeroMQ 广播通信地址
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:       # 返回调度器允许的最大前向长度（max_forward_len）
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:  # 返回调度器是否需要创建解码器链接，默认情况下为 True
        return True
