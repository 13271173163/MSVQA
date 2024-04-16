import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        # 隐藏层的维度
        self.hidden = hidden
        # Transformer块的数量
        self.n_layers = n_layers
        # 注意力头的数量
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # 调用BERTEmbedding类创建一个嵌入层self.embedding
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        # nn.ModuleList创建一个包含多个transformer块的列表，每个transformer块由TransformerBlock类实现,每个块都相同
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    # x和segment_info分别表示，数字序列化后的token和segment_label
    # segment_info是分段信息，用于指示每个单词所属的段落
    def forward(self, x, segment_info):
        # attention masking for padded token
        # mask的维度为torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # 该语句是为了构造掩码矩阵，让计算自注意力矩阵的时候，消除默认填充值0的影响。
        # 构造token时，设置的self.pad_index = 0
        # 为了消除填充值0带来的影响，必须将填充值0都替换为极小的负值。因为这样经过softmax之后会无限趋近于0。
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
