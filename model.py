import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,n_embed,n_hidden,n_label,batch_size):
        super(Model,self).__init__()
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_label = n_label
        self.embed = nn.Embedding(self.n_embed,self.n_hidden)
        self.linear = nn.Linear(self.n_hidden,self.n_label)
        self.batch_size = batch_size

    def forward(self,x):
        x = self.embed(x)

        x = F.max_pool1d(x.permute(0, 2, 1), x.size()[1])
        logit = self.linear(x.view(x.size()[0], self.n_hidden))
        return logit
