from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab  # 词汇表,包含模型的词汇信息
        self.seq_len = seq_len  # 序列的长度,代表输入数据的最大长度
        self.on_memory = on_memory  # 是否将整个语料库存储在内存中,默认为true
        self.corpus_lines = corpus_lines  # 数据库文件的行数
        self.corpus_path = corpus_path   # corpus_path:语料库(数据)的路径;
        self.encoding = encoding  # 数据的编码格式

        with open(corpus_path, "r", encoding=encoding) as f:
            # on_memory为false时,打开语料库文件逐行读取,并通过循环计数器self.corpus_lines统计文件行数
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1
            
            # on_memory为true时,将整个语料库文件加载到内存中
            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            # 顺序读取语料
            self.file = open(corpus_path, "r", encoding=encoding)
            # 随机读取语料
            self.random_file = open(corpus_path, "r", encoding=encoding)
            # 通过循环,随即跳过一定行数(最多1000行)来实现随机读取文件内容的效果
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    # 返回语料的长度
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 为NSP任务构造样本
        t1, t2, is_next_label = self.random_sent(item)
        # 为Mask任务构造样本
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # 为样本添加特殊标签[SEP]  [CLS]
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        # 构造segment-label,第一句对应1,第二句对应2
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        # 对长度超过bert最大序列长度进行截断,对短的序列进行填充补齐以达到seq_len最大长度
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    # 为MASK任务制造样本,85%的样本不做MASK,保持原样,同时样本的标签为0
    # 另外15%中,80%做MASK,10%随即替换成别的词,10%保持原样,同时标签为字符对应的数字
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    # 该函数50%概率返回索引到的样本的第一个句子和第二个句子，50%的概率返回索引到的样本的第一句话和随机选择一个样本的第二句话
    # 为NSP构造样本用的
    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    # 该函数实现分别返回索引后的样本的两个句子,比如索引第六条文本,第六条文本中的两个句子就会被存储在两个变量中然后返回
    # 如果语料被加载到内存中时,可以通过self.lines[item]来获取指定行的内容,item代表文本中的第几行
    # 如果语料没有加载到内存,则采用逐行读取,self.file.__next__()用于读取下一行
    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    # 随机选择一行样本,并返回第二句话
    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
