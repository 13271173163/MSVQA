import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # 词嵌入层，vocab_size表示词汇表大小，embed_size词嵌入向量的维度
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # PositionalEmbedding类创建一个位置编码层self.position，用于为输入序列的每个位置添加位置信息。
        # d_model，表示Transformer模型的隐藏层维度。
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        # SegmentEmbedding类创建一个段落编码层self.segment，用于区分输入序列中不同段落的内容。
        # embed_size，与词嵌入向量的维度保持一致。
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        # dropout防止过拟合，增强模型的泛化能力
        # nn.Dropout类创建一个丢弃层self.dropout，用于在模型训练过程中随机丢弃一部分神经元，以防止过拟合。
        self.embed_size = embed_size

        # sequence是一个文本序列的张量表示，其形状为(batch_size, seq_len)，其中batch_size表示批次大小，seq_len表示序列长度。
        # segment_label是一个张量表示各个文本序列中的段落标签，其形状也为(batch_size, seq_len)
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
