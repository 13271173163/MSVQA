import torch.nn as nn

from .bert import BERT

# 综合后面两个任务，实施预训练
class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)

# 实现NSP预训练任务计算逻辑
class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        # NSP任务是将CLS向量取出来做二分类
        # hidden参数表示BERT模型的输出大小。
        # 将BERT模型输出的向量映射到2维，分别表示是否是连续的句子和非连续的句子。
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

# 实现Mask预训练任务计算逻辑
class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        # MASK任务是将整个向量矩阵做多分类，类别数就是词表大小。
        # hidden参数表示BERT模型的输出大小，vocab_size参数表示总词汇表的大小。
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        # 输出每个词汇在当前位置的概率分布

    def forward(self, x):
        return self.softmax(self.linear(x))
