from classifier import *

train = Train()
word_alpha,label_alpha = train.create_alphabet('data/train.fmt')
# print(word_alpha.dict)
# print(label_alpha.dict)
examp1 = train.create_feature('data/train.fmt',word_alpha,label_alpha)
examp2 = train.create_feature('data/dev.fmt',word_alpha,label_alpha)
examp3 = train.create_feature('data/test.fmt',word_alpha,label_alpha)

# for i in examp1:
#     i.show()
# for i in examp2:
#     i.show()

train.train(examp1,15,len(word_alpha.dict),50,len(label_alpha.dict),examp2,examp3)
