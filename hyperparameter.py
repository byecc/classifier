class HyperParameter:

    def __init__(self):
        self.batch_size=50  #require above zero
        self.epoch = 100
        self.n_embed = 0
        self.n_hidden = 0
        self.n_label = 0
        self.learn_rate = 0.1

        self.train_file = 'data/train.fmt'
        self.dev_file = 'data/dev.fmt'
        self.test_file = 'data/test.fmt'

        self.padding_index = 0

    def add_layer_num(self,n_embde,n_hidden,n_label):
        self.n_embed = n_embde
        self.n_hidden = n_hidden
        self.n_label = n_label

    def add_word_index(self,padding_index):
        self.padding_index = padding_index


