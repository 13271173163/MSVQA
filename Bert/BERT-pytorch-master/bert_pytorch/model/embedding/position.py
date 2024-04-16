import torch.nn as nn
import torch
import math

# 实现位置编码的功能,为输入序列每个位置添加位置信息
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        "创建了一个形状为(max_len, d_model)的零张量pe来存储位置编码。"
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        "通过torch.arange函数生成一个长度为max_len的浮点数序列，并使用unsqueeze(1)方法将其从形状为(max_len,)变为(max_len, 1)的张量，表示每个位置的索引。"
        position = torch.arange(0, max_len).float().unsqueeze(1)
        "torch.arange函数再次生成一个长度为d_model的浮点数序列，并乘以一个比例常数-(math.log(10000.0) / d_model)，最后使用exp()方法计算得到div_term。"
        "div_term的作用是用于计算正弦和余弦部分的角度。"
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        "对位置编码张量pe进行切片操作,将奇数和偶数位置的元素分别赋值为正弦和余弦函数的值，其中使用了广播机制。"
        "pe[:, 0::2]表示从第0列开始，步长为2地选择列进行赋值；"
        pe[:, 0::2] = torch.sin(position * div_term)
        "pe[:, 1::2]表示从第1列开始，步长为2地选择列进行赋值。"
        pe[:, 1::2] = torch.cos(position * div_term)
        "这样就将位置编码的正弦和余弦部分计算出来并赋值给了pe。"

        "unsqueeze(0)方法将pe的形状从(max_len, d_model)变为(1, max_len, d_model)，并使用register_buffer方法将其声明为模型的一个缓冲区。"
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        "这样，在模型运行过程中，该位置编码张量将被保留，并且不会作为模型参数进行训练。"

    "输入x代表模型的输入，它的形状为(batch_size, seq_len, d_model)。"
    def forward(self, x):
        return self.pe[:, :x.size(1)]
    "在这里，通过切片操作self.pe[:, :x.size(1)]从位置编码张量中取出与输入序列长度相对应的部分，这样就得到了与输入序列相匹配的位置编码张量。最后，将该位置编码张量作为输出返回。"
