from classifier import *
from hyperparameter import *
import torch

torch.manual_seed(1)

parameter = HyperParameter()
train = Train()

embed_dict = train.load_word_embedding(parameter.wordEmbed_file)

word_alpha,label_alpha,we = train.create_alphabet(parameter.train_file,embed_dict)

# print(word_alpha.dict)
# print(label_alpha.dict)

examp1 = train.create_feature(parameter.train_file,word_alpha,label_alpha)
examp2 = train.create_feature(parameter.dev_file,word_alpha,label_alpha)
examp3 = train.create_feature(parameter.test_file,word_alpha,label_alpha)

parameter.add_layer_num(len(word_alpha.dict),300,len(label_alpha.dict))
parameter.add_param(word_alpha.dict['-padding-'],we)
# for i in examp1:
#     i.show()
# for i in examp2:
#     i.show()

train.train(parameter,examp1,examp2,examp3)
