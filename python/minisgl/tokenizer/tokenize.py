from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase

# 处理文本分词：主要作用是将输入的文本信息使用 Hugging Face 的分词器转换为 PyTorch 张量
class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer          # 来自 Hugging Face transformers 库的预训练分词器

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):  # 如果是列表 (List): 通常代表这是一种多轮对话格式
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,             # 只拼接文本暂不分词
                    add_generation_prompt=True, # 自动在末尾加上助手（Assistant）的起始标记，引导模型开始生成回复
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            # 将拼接好的字符串 prompt 转换为 PyTorch 格式的 Token ID 张量
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")  
            )
            results.append(input_ids.view(-1).to(torch.int32))  
        return results
