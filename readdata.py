import re
import random
import math

class  Instance:
    def __init__(self):
        self.word = []
        self.label =''

    def show(self):
         print(self.word, self.label)

class Read_Data:
    def __init__(self,path):
        self.path = path

    def clean_str(self,string):
        """
                Tokenization/string cleaning for all datasets except for SST.
                Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
                """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def process_twitter_task(self):
        result=[]
        with open(self.path,'r') as f:
            for line in f.readlines():
                info = Instance()
                sentence = line.strip().split('|||')
                data = self.clean_str(sentence[0].strip()).split(' ')
                info.word = data
                info.label = sentence[1].strip()
                result.append(info)
        # for i in result:
        #     print(i.word,i.label)
        return result

    def process_new_task(self):
        result = []
        with open(self.path,'r') as f:
            for line in f.readlines():
                info = Instance()
                sentence = line.strip().split(' ',1)
                info.label = sentence[0]
                info.word = sentence[1].split(' ')
                result.append(info)
        return result

    def generate_new_data(self,path):
        data_list = []
        with open(path,'r') as f:
            for line in f.readlines():
                data_list.append(line)
        random.shuffle(data_list)
        for i in range(len(data_list)):
            if i <= round(len(data_list)*0.7):
                with open(path+'.train','a') as f:
                    f.write(data_list[i])
            elif round(len(data_list))*0.7<i<= round(len(data_list)*0.8):
                with open(path+'.dev','a') as f:
                    f.write(data_list[i])
            else:
                with open(path+'.test','a') as f:
                    f.write(data_list[i])
