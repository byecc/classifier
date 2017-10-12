from instance import *
from readdata import *
import torch
import torch.nn.functional as F
from model import Model
import time

class Train:

    def create_alphabet(self,data_file):
        rd = Read_Data(data_file)
        result = rd.process()
        word_alpha = Alphabet()
        label_alpha = Alphabet()
        for i in result:
            word_alpha.create(i.word)
            label_alpha.create(i.label)
        word_alpha.list.append('-unknown-')
        word_alpha.dict['-unknown-'] = word_alpha.index
        label_alpha.list.append('-unknown-')
        label_alpha.dict['-unknown-'] = label_alpha.index
        # print(word_alpha.dict,label_alpha.dict)
        return word_alpha,label_alpha

    def create_feature(self,data_file,word_alpha,label_alpha):
        rd = Read_Data(data_file)
        result = rd.process()
        examples = []
        for i in result:
            feat = Feature()
            ex = Example()
            for word in i.word:
                if word in word_alpha.dict.keys():
                    feat.word_index.append(word_alpha.dict[word])
                else:
                    feat.word_index.append(len(word_alpha.dict)-1)
                feat.word_len.append(len(word))
            ex.feature = feat
            ex.label_index=label_alpha.dict[i.label]
            examples.append(ex)
        return examples

    def toVariables(self,example):

        x = torch.autograd.Variable(torch.LongTensor([example.feature.word_index]))
        y = torch.autograd.Variable(torch.LongTensor([example.label_index]))
        # x.data[0] = example.feature.word_index
        # y.data[0]=example.label_index
        return x,y

    def train(self,examples,circulate_num,n_embed,n_hidden,n_label,ex2,ex3):
        """

        :param examples: 训练的样本
        :param circulate_num: 训练次数
        :param n_embed: input layer num
        :param n_hidden: hidden layer num
        :param n_label: output layer num
        :return:
        """
        model = Model(n_embed,n_hidden,n_label)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.05)
        for i in range(circulate_num):
            print("第%d次:"%(i+1))
            starttime = time.time()
            sum=0
            correct=0
            loss_sum = 0.0
            for example in examples:
                optimizer.zero_grad()
                x,y = self.toVariables(example)
                logit = model.forward(x)
                # torch.nn.MSELoss
                loss = F.cross_entropy(logit,y)
                loss.backward()
                optimizer.step()

                if y.data[0]==self.getMaxIndex(logit):
                    correct+=1
                sum+=1
                loss_sum += loss.data[0]
            print("loss:",(loss_sum/sum))
            print('train accuracy :',(correct/sum))
            endtime = time.time()
            print('after',(endtime-starttime),'s')
        cor=0
        s=0
        for ex in ex2:
            x2, y2 = self.toVariables(ex)
            logit = model.forward(x2)
            if y2.data[0] == self.getMaxIndex(logit):
                cor += 1
            s += 1
        print('ex2 accuracy:',(cor/s))



    def getMaxIndex(self,score):    #获取最大权重的下标
        label_size=score.size()[1]
        maxIndex=0
        max=score.data[0][0]
        for idx in range(1,label_size):
            tmp=score.data[0][idx]
            if max<tmp:
                max=tmp
                maxIndex=idx
        return maxIndex







