import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

import tqdm

# 训练代码。包括读取数据、加载模型、设定优化器、计算损失函数、分布式GPU、一轮轮迭代的具体逻辑、保存模型。
class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train 你想要训练的bert模型
        :param vocab_size: total word vocab size 总的词汇表大小
        :param train_dataloader: train dataset data loader 训练数据下载
        :param test_dataloader: test dataset data loader [can be None] 测试数据下载
        :param lr: learning rate of optimizer 学习率
        :param betas: Adam optimizer betas 正则化时设置的参数
        :param weight_decay: Adam optimizer weight decay param  正则化时的权重
        :param with_cuda: training with cuda 是否用GPU
        :param log_freq: logging frequency of the batch iteration  批次吧
        """
        # 值得注意的是 在模型创建时，在构造函数中使用.to(device)方法将模型参数移动到CUDA设备上。

        "判断是否有可用的cuda"
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        "每个epoch都会保存模型参数到cuda设备上"
        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        "加载训练和测试数据"
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        "设置Adam优化器"
        "ScheduledOptim类，它是一个自定义的学习率调整器。在BERT模型的训练过程中，需要逐渐降低学习率。"
        "传入Adam优化器、BERT模型的输出大小（即hidden_size），以及warmup_steps参数，其中warmup_steps表示调整学习率的步数。"
        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        "设置损失函数"
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        """
        enumerate(data_loader)将数据加载器包装成一个可迭代对象，并使用enumerate()函数对其进行枚举，返回索引和对应的数据批次。

        desc参数是进度条的描述信息，格式为"EP_XXX:YYY"，其中XXX表示当前的训练阶段（例如"train"或"valid"），YYY表示当前的训练轮数（即epoch）。

        total参数设置进度条的总长度，即数据加载器中的批次数量。

        bar_format参数设置进度条的格式，"{l_bar}{r_bar}"表示左侧显示进度信息，右侧显示剩余时间。
        """

        avg_loss = 0.0 # 平均损失
        total_correct = 0 # 正确预测的数量
        total_element = 0 # 一共预测的数据的数量

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss
            # 利用交叉熵计算，两个任务损失值相加得到总的损失。

            # 3. backward and optimization only in train
            "在训练模式下，进行反向传播"
            if train:
                "self.optim_schedule.zero_grad()将优化器中所有参数的梯度置零，以便进行下一次反向传播。"
                self.optim_schedule.zero_grad()
                "loss.backward()根据当前批次的损失值，计算参数的梯度。通过调用backward()函数，反向传播算法会自动计算每个参数的梯度，并将它们保存在相应的参数对象的.grad属性中。"
                loss.backward()
                "self.optim_schedule.step_and_update_lr()根据梯度更新参数，并调整学习率。"
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    # 保存模型参数数据，在一个指定文件中。
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
