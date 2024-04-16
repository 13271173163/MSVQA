import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)
"""
将embed_size的默认值设为512，并将该embedding层的词表大小设置为3。
至于词表为何大小为3呢？第一句的编码全为1，第二句的编码全为2，还有如果不够长的补齐编码是0，正好是3。

"""
