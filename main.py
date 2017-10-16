from classifier import *
from hyperparameter import *
import torch

torch.manual_seed(1)

parameter = HyperParameter()
train = Train()

word_alpha,label_alpha = train.create_alphabet(parameter.train_file)

# print(word_alpha.dict)
# print(label_alpha.dict)

examp1 = train.create_feature(parameter.train_file,word_alpha,label_alpha)
examp2 = train.create_feature(parameter.dev_file,word_alpha,label_alpha)
examp3 = train.create_feature(parameter.test_file,word_alpha,label_alpha)

parameter.add_layer_num(len(word_alpha.dict),50,len(label_alpha.dict))
parameter.add_word_index(word_alpha.dict['-padding-'])
# for i in examp1:
#     i.show()
# for i in examp2:
#     i.show()

train.train(parameter,examp1,examp2,examp3)
