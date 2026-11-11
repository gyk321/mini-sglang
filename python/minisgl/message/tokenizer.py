from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg: # 基础消息类
    @staticmethod       # 静态方法
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)                  # 将当前的 Python 对象转换为字典

    @staticmethod       # 静态方法
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)    # 接收一个字典并调用 deserialize_type 将其还原为对应的 BaseTokenizerMsg 子类对象

# 在 LLM 推理中，为了提高 GPU 的利用率和吞吐量，系统通常会将多个用户的请求打包成一个 Batch 一起处理
@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):          # 批量消息类
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):              # 反分词/解码消息类，将模型生成的 Token ID 转换回人类可读的文本字符串
    uid: int                                        # 请求的唯一标识符（整型），用于追踪是哪个用户的请求
    next_token: int                                 # 模型刚刚生成的下一个 Token 的 ID
    finished: bool                                  # 用于标记当前请求的文本生成是否已经结束


@dataclass
class TokenizeMsg(BaseTokenizerMsg):                # 分词消息类，用于接收用户的输入并准备让模型进行推理
    uid: int                                        # 请求的唯一标识符
    text: str | List[Dict[str, str]]                # 用户的输入内容。它支持两种格式：普通的字符串 (str)，或者类似于 OpenAI API 的对话列表
    sampling_params: SamplingParams                 # 采样参数，包含了控制生成文本的随机性和多样性的各种设置，如温度、Top-k、Top-p 等


@dataclass
class AbortMsg(BaseTokenizerMsg):                   # 中止消息类，用于通知系统终止某个正在进行的请求
    uid: int                                        # 只需要传递要取消的请求的唯一标识符
