import torch.nn.functional as F
import torch.nn as nn
import torch.autograd
import numpy as np
import random

torch.manual_seed(23)

class Model(nn.Module):
    def __init__(self,hyperparameter):
        super(Model,self).__init__()
        self.param = hyperparameter
        self.n_embed = hyperparameter.n_embed
        self.embed_dim = hyperparameter.embed_dim
        self.n_label = hyperparameter.n_label
        self.embed = nn.Embedding(self.n_embed,self.embed_dim)
        # self.cnn = nn.Conv1d(hyperparameter.cnn_in_channels,
        #                       hyperparameter.cnn_out_channels,
        #                       hyperparameter.cnn_kernel_size,
        #                       hyperparameter.cnn_stride)
        self.linear = nn.Linear(self.embed_dim,self.n_label)

        self.batch_size = hyperparameter.batch_size

    def pretrain(self,teso):
        nump = []
        for i in teso:
            array = []
            k=0
            for j in i:
                array.append(float(j))
                k+=1
            while k<=300:
                # array.append(random.uniform[-0.25,0.25])
                k+=1
            nump.append(array)
        self.embed.weight.data.copy_(torch.Tensor(nump))

    def forward(self,x):
        x = self.embed(x)
        # x = self.cnn(x)
        x = F.max_pool1d(x.permute(0, 2, 1), x.size()[1])
        logit = self.linear(x.view(x.size()[0],self.embed_dim))
        return logit
