from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP

# 张量并行（Tensor Parallelism, TP）在 Embedding 层的标准实现方式（沿词表维度即行维度切分权重，类似于 Megatron-LM 的设计）
class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)     # 将大小为 num_embeddings（词表大小 V）的权重矩阵按行（词表维度）均匀切分到 tp_size 个 GPU 上
        start_idx = self.num_embeddings_tp * tp_rank                        # GPU（tp_rank）负责的全局 Token ID 起点
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)    # 防止最后一个 Rank 越过真实词表上限 V
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)    # 每个 GPU 仅维护形状为 (num_embeddings_tp, embedding_dim) 的部分权重。
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        y = indexing(   # 掩码查表
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y          # 全局求和规约 将所有 GPU 上的输出张量 y 按元素相加

# 大语言模型输出层（Language Model Head）的张量并行（Tensor Parallelism, TP）实现
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,                     # 每个 GPU 同样只分配对应自身词表分片大小的偏置向量
        tie_word_embeddings: bool = False,      # 权重共享
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],        # 包含模型权重的字典
        *,
        prefix: str = "",                           # 当前模块在整个模型树中的前缀路径
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:     # 未启用权重共享
            # 从 state_dict 中读取当前 TP Rank（张量并行分片）对应的权重
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:                           # 启用了权重共享
            # pop the lm_head.weights and lm_head.bias if they exist
            # 真正的数据加载动作已经由输入层（VocabParallelEmbedding）完成了
            # 为了避免重复加载，直接从 state_dict 中删除 lm_head 的权重和偏置
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",               # 当前模块在模型中的前缀路径
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:     # 未启用权重共享（独立权重）
            return super().state_dict(prefix=prefix, result=result)     # 将当前 GPU 负责的 self.weight（和 self.bias）切片正常打包到字典中并返回
        return {} if result is None else result     # 为了避免在同一个 Checkpoint 文件中把同一份数据保存两次，这里直接阻断了 LM Head 权重的导出
    # 将 Transformer 最后一层的隐藏状态（Hidden States）映射为预测下一个词的概率分布
    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()          # 获取上下文与预填充（Prefill）优化
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)  # 提取出这 bs（Batch Size）个末尾 Token 的索引
            x = x[indices].contiguous() # 对 x 进行切片取值
            del indices

        module = self.tied_embedding or self        # 如果有绑定的输入层权重（self.tied_embedding），就复用它的权重，否则使用自身的 self.weight
        logits = F.linear(x, module.weight, self.bias)      # 由于采用了张量并行，得到的 logits 是局部的，形状为 [bs, num_embeddings_tp]
        if self.tp_size == 1:           # 没有开启张量并行
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)       # 调用底层的 NCCL 通信原语 all_gather

        if bs == 1:                     # 内存顺序天然就是正确的词表拼接顺序
            return output_tensor.view(1, -1)[:, : self.num_embeddings]
        # 将一维堆叠的张量还原为 3D 视图：[tp_size, bs, num_embeddings_tp]
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()     # 交换前两个维度，变成 [bs, tp_size, num_embeddings_tp]
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))   # 将后两个维度合并，得到 [bs, tp_size * num_embeddings_tp]
        return output_tensor[:, : self.num_embeddings]
