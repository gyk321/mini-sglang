from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]

# DetokenizeManager 通过维护滑动上下文窗口解决了子词上下文丢失（前缀空格问题）和多字节 UTF-8 字符截断（乱码问题）
@dataclass
class DecodeStatus:
    decoded_ids: List[int]          # 当前请求已生成的所有 token ID 列表
    decoded_str: str                # 已经确认完整且无乱码的累计解码字符串
    read_offset: int  # length of read ids          上一次成功完成完整字符解码时的 decoded_ids 长度
    surr_offset: int  # length of surr ids          用来提供解码前缀，防止丢失前导空格
    sent_offset: int  # length of sent out string   已经发送给客户端的字符串长度

# 流式增量解码（Incremental Detokenization）的核心管理类
# 在大模型逐个生成 token 的过程中，将 token ID 实时、准确地转换为可读文本，
# 同时妥善处理子词拼接空格（如 SentencePiece 的前导空格）和多字节 UTF-8 字符（如中文、Emoji）被截断的问题
class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}       # 跟踪每个并发请求的状态
        self.tokenizer = tokenizer                          # HuggingFace 分词器，用于执行实际的解码操作
        self.eos_token_id = self.tokenizer.eos_token_id     # 结束符 token 的 ID，用于判断序列是否结束

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []              # 从 surr_offset 到最新的末尾，这是包含“已知前缀 + 新 token”的序列
        surr_ids: List[List[int]] = []              # 从 surr_offset 到 read_offset，这纯粹是“已知前缀”的序列
        for msg in msgs:                            # 遍历传入的请求消息列表
            if msg.uid not in self.decode_map:      # 遇到新的请求
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],         # 该请求目前已解码的所有 token ID
                    decoded_str="",         # 迄今累积的已确认、完全解码的文本
                    read_offset=0,          # 已经确认解码为完整字符/单词的 token 索引。
                    surr_offset=0,          # 历史上下文索引（通常等于上一次的 read_offset）
                    sent_offset=0,          # 已经发送/返回给用户的字符串长度，用于计算每次的增量输出
                )
            s = self.decode_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):  # 如果当前生成的不是结束符
                s.decoded_ids.append(msg.next_token)                        # 把新的 next_token 加入到该请求的历史 token 列表中
            read_ids.append(s.decoded_ids[s.surr_offset :])                 # 当前待解码整体,包含“最近已确认的安全上下文” + “刚生成的新 token”。
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])   # 上下文前缀:代表已经确认解码成功的一小段历史 token,只包含“最近已确认的安全上下文”（作为前缀基准）。

        # 利用 HuggingFace tokenizer 的 batch_decode，一次性将所有请求的 read_ids 和 surr_ids 转换为字符串
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            # 通过字符串相减（去掉历史前缀的字符串 surr_str），提取出真正属于本次新生成的文本
            new_text = read_str[len(surr_str) :]
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):        # 完整无截断，新文本不以未知字符 `` 结尾
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:                                                        # 截断/多字节未拼完
                new_text = find_printable_text(new_text)                # 它会将乱码部分砍掉，只保留前面已经打印完整的部分
                output_str = s.decoded_str + new_text
            # 利用 sent_offset（已经发送给客户端的字符长度）从 output_str 中切片，得到本次真正需要发往客户端的新字符串
            incremental_output = output_str[s.sent_offset :]            # 截取增量
            s.sent_offset = len(output_str)                             # 更新 sent_offset 为下一次计算做准备
            incremental_strs.append(incremental_output)
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
