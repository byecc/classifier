class Alphabet:
    def __init__(self):
        self.list=[]
        self.dict={}
        self.index = 0

    def create(self,word_data):
        for word in word_data:
            if word not in self.list:
                self.list.append(word)
                self.dict[word]=self.index
                self.index+=1

    def show(self):
        for i,j in self.dict.items():
            print(i,j,end=' ')

class Feature:
    def __init__(self):
        self.word_index=[]
        self.word_len=[]

class Example:
    def __init__(self):
        self.feature=Feature()
        self.label_index=-1

    def show(self):
        print(self.feature.word_index,self.label_index)