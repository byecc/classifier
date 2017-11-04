from instance import *
from readdata import *
import torch
import torch.autograd
import torch.nn.functional as F
from model import Model
import time
import torch.utils.data as Data
import random
import numpy as np

random.seed(21)
class Train:

    def create_alphabet(self,data_file,embed_dict):
        rd = Read_Data(data_file)
        # result = rd.process_twitter_task()
        result = rd.process_new_task()
        word_alpha = Alphabet()
        label_alpha = Alphabet()
        for i in result:
            word_alpha.create(i.word)
            label_alpha.create(i.label)
        word_alpha.dict['-unknown-'] = word_alpha.index
        word_alpha.dict['-padding-']=word_alpha.index+1
        # label_alpha.dict['-unknown-'] = label_alpha.index
        # print(word_alpha.dict,label_alpha.dict)
        we = []
        for i in word_alpha.dict.keys():
            if i in embed_dict.keys():
                we.append(embed_dict[i])
                # print(i,embed_dict[i])
            else:
                we.append([random.uniform(-0.25,0.25) for i in range(300)]) # file : converted_word....txt
                # we.append([random.uniform(-0.25,0.25) for i in range(100)]) # file : w2v103100-en
        return word_alpha,label_alpha,we

    def create_feature(self,data_file,word_alpha,label_alpha):
        rd = Read_Data(data_file)
        # result = rd.process_twitter_task()
        result = rd.process_new_task()
        examples = []
        for i in result:
            feat = Feature()
            ex = Example()
            for word in i.word:
                if word in word_alpha.dict.keys():
                    feat.word_index.append(word_alpha.dict[word])
                else:
                    feat.word_index.append(word_alpha.dict['-unknown-'])
                feat.word_len +=1
            ex.feature = feat
            ex.label_index=label_alpha.dict[i.label]
            examples.append(ex)
        return examples

    def toVariables(self,example):
        x = torch.autograd.Variable(torch.LongTensor([example.feature.word_index]))
        y = torch.autograd.Variable(torch.LongTensor([example.label_index]))
        return x,y

    # def create_batch(self):

    def create_batchdata(self,ex,pad_ix):
        max_sentence_size = 0
        batch_num = len(ex)
        for e in ex:
            if max_sentence_size < e.feature.word_len:
                max_sentence_size = e.feature.word_len
        batch_input_unify = torch.autograd.Variable(torch.LongTensor(batch_num,max_sentence_size))
        batch_label_unify = torch.autograd.Variable(torch.LongTensor(batch_num))

        for i in range(batch_num):
            example =ex[i]
            batch_label_unify.data[i] = example.label_index
            for j in range(max_sentence_size):
                if j < len(example.feature.word_index):
                    batch_input_unify.data[i][j] = example.feature.word_index[j]
                else:
                    batch_input_unify.data[i][j] = pad_ix
        return batch_input_unify,batch_label_unify

    def batch_toVariable(self,batch_input,batch_label,ix_list,size):
        test1 = batch_input.data.numpy()
        test2 = batch_label.data.numpy()
        batch_block_list = [np.ndarray.tolist(np.array(test1[ix_list[i]])) for i in range(size)]
        batch_label_list = [np.ndarray.tolist(np.array(test2[ix_list[i]])) for i in range(size)]
        # batch_block_list = []
        # batch_label_list = []
        # for i in range(size):
        #     batch_block_list.append(np.ndarray.tolist(np.array(test1[ix_list[i]])))
        #     batch_label_list.append(np.ndarray.tolist(np.array(test2[ix_list[i]])))
        x = torch.autograd.Variable(torch.LongTensor(batch_block_list))
        y = torch.autograd.Variable(torch.LongTensor(batch_label_list))

        for i in range(size):
            ix_list.pop(0)
        return x,y

    def train(self,parameter,ex1,ex2,ex3):
        """
        :param parameter: hyperparameter
        :param ex1: train examples
        :param ex2: dev examples
        :param ex3: test examples
        :return:
        """
        batch_block = len(ex1)//parameter.batch_size
        remain = len(ex1)%parameter.batch_size
        if remain != 0:
            batch_block +=1
        model = Model(parameter)
        print(model)
        model.pretrain(parameter.word_embed)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.05)
        batch_input_unify,batch_label_unify = self.create_batchdata(ex1,parameter.padding_index)
        sentence_list = []
        stime = time.time()
        for i in range(len(ex1)):
            sentence_list.append(i)
        for i in range(parameter.epoch):
            print("第%d次:"%(i+1))
            starttime = time.time()
            sum=0
            correct=0
            loss_sum = 0.0
            # for example in ex1:      # 未加batch
            #     optimizer.zero_grad()
            #     x,y = self.toVariables(example)
            #     logit = model.forward(x)
            #     # torch.nn.MSELoss
            #     loss = F.cross_entropy(logit,y)
            #     loss.backward()
            #     optimizer.step()
            sen_idx = [s for s in sentence_list]
            random.shuffle(sen_idx)
            for block in range(batch_block):
                optimizer.zero_grad()
                if block ==batch_block-1:
                    x,y = self.batch_toVariable(batch_input_unify,batch_label_unify,sen_idx,remain)
                else:
                    x,y = self.batch_toVariable(batch_input_unify,batch_label_unify,sen_idx,parameter.batch_size)
                logit = model.forward(x)
                loss = F.cross_entropy(logit,y)
                loss.backward()
                optimizer.step()
                if y.data[0]==self.getMaxIndex(logit):
                    correct+=1
                sum+=1
                loss_sum += loss.data[0]
            print("loss:",(loss_sum/sum))
            accuracy = correct/sum
            print('train accuracy :',accuracy)
            endtime = time.time()
            print('after',(endtime-starttime),'s')
            if accuracy==1.0:
                break
            self.eval(model,ex2,'dev')
        self.eval(model,ex2,'dev')
        self.eval(model,ex3,'test')

    def eval(self,model,eval_data,data_name):
        cor = 0
        s = 0
        for e in eval_data:
            x,y = self.toVariables(e)
            logit = model(x)
            if y.data[0]==self.getMaxIndex(logit):
                cor+=1
            s+=1
        print(data_name +' accuracy:',(cor/s))

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

    def load_word_embedding(self,path):
        dict={}
        with open(path,'r') as f:
            fr = f.readlines()
            embed_dim = fr[0] # file: data/converted_word_Subj.txt; data/converted_word_CR.txt
            # embed_dim = fr[0].strip().split(' ')[1] # file: data/w2v103100-en
            for line in fr[1:]:
                wordv = line.strip().split(' ',1)
                dict[wordv[0]] = wordv[1].split(' ')
        dict['-unkwon-'] = [random.uniform(-0.25,0.25) for i in range(int(embed_dim))]
        dict['-padding-'] = [random.uniform(-0.25,0.25) for i in range(int(embed_dim))]
        return dict







