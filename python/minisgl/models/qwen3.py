from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        # 初始化解码器层
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)      # 自注意力模块
        self.mlp = Qwen3MLP(config)                     # 前馈神经网络模块
        self.input_layernorm = RMSNormFused(            # 注意力层之前的均方根归一化层
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(   # MLP 层之前的均方根归一化层
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id                       # 记录当前层的编号，供性能分析工具使用

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")                  # 方便开发者在使用 Nsight Systems 等工具进行 GPU 性能剖析
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)             # 注意力前归一化与残差融合
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)    # MLP 前归一化与残差融合
        x = self.mlp.forward(x)                                             # 前馈网络计算
        return x, residual  


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(     # 词嵌入层
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(                           # 解码器层列表
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(                       # 最终归一化层
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)        # 将输入的张量（包含 token IDs）通过词嵌入层转换为初始的隐藏状态张量
        residual: torch.Tensor | None = None            # 残差连接初始化
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)    # 依次穿过每一层解码器,不断更新当前的特征状态和残差
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(                          # 并行输出头
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,     # 词嵌入权重共享
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 从推理引擎的全局执行上下文（Global Context）中提取当前批次的 Token ID
        # 便于统一管理连续批处理（Continuous Batching）、KV Cache 页表和 CUDA Graph 捕获。
        output = self.model.forward(get_global_ctx().batch.input_ids)

        # 计算并返回未归一化的词表预测得分 logits
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLM"]
