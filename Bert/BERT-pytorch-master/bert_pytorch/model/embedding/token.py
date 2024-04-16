import torch.nn as nn

# 继承nn.Embedding,并做了一个拓展,将embed_size设置为512
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
# padding_idx的作用:
# 如果被指定,则padding_idx上的项不会参与梯度更新,即它仍然是一个固定的"pad",对于一个新构造的Embedding,在padding_idx下的嵌入向量将默认全为零,但可以更新为另一个值用作填充向量
# 举例说明
"""
import torch
import torch.nn as nn

embed = nn.Embedding(10, 3, padding_idx=3)  # padding_idx 默认是0
x = torch.tensor([[2, 2, 4, 3], [1, 2, 5, 4]])
print(embed(x))

输出为：
tensor([[[-1.6093,  0.3152, -1.4056],
         [-1.6093,  0.3152, -1.4056],
         [-1.0126,  0.4661, -0.7098],
         [ 0.0000,  0.0000,  0.0000]],

        [[ 0.3183,  0.7932,  0.0737],
         [-1.6093,  0.3152, -1.4056],
         [-0.0623, -0.7012,  0.4782],
         [-1.0126,  0.4661, -0.7098]]], grad_fn=<EmbeddingBackward>)
2*4*3的向量，其中输入向量中id为3的那个向量对应的行全为0。
如果x改变一下
x = torch.tensor([[3, 2, 4, 3], [1, 2, 5, 4]])
输出随之改变：
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.9111,  1.3400,  0.1826],
         [-1.6813,  2.2275,  0.4838],
         [ 0.0000,  0.0000,  0.0000]],

        [[-0.4771, -1.0103, -0.8225],
         [ 0.9111,  1.3400,  0.1826],
         [ 0.3259, -0.1886, -0.3815],
         [-1.6813,  2.2275,  0.4838]]], grad_fn=<EmbeddingBackward>)
应该明白padding_idx的作用了。因为本工程所有补齐的填充值默认都是0，所以该功能可以让填充值对权重更新不产生影响。
"""
