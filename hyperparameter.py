class HyperParameter:

    def __init__(self):
        self.batch_size=50  #require above zero
        self.epoch = 100
        self.n_embed = 0
        self.embed_dim = 0
        self.n_label = 0
        self.learn_rate = 0.1
        self.cnn_in_channels = 47
        self.cnn_out_channels = 47
        self.cnn_kernel_size = 50
        self.cnn_stride = 1

        #twitter_task
        # self.train_file = 'data/train.fmt'
        # self.dev_file = 'data/dev.fmt'
        # self.test_file = 'data/test.fmt'
        # subj
        self.train_file = 'data/subj.all.train'
        self.dev_file = 'data/subj.all.dev'
        self.test_file = 'data/subj.all.test'
        # CR
        # self.train_file = 'data/custrev.all.train'
        # self.dev_file = 'data/custrev.all.dev'
        # self.test_file = 'data/custrev.all.test'

        self.wordEmbed_file='data/converted_word_Subj.txt'

        self.padding_index = 0
        self.word_embed = []

    def add_layer_num(self,n_embde,embed_dim,n_label):
        self.n_embed = n_embde
        self.embed_dim = embed_dim
        self.n_label = n_label

    def add_param(self,padding_index,word_embed):
        self.padding_index = padding_index
        self.word_embed = word_embed

    def add_cnn_param(self,in_channels,out_channels,kernel_size,stride):
        self.cnn_in_channels = in_channels
        self.cnn_out_channels = out_channels
        self.cnn_kernel_size = kernel_size
        self.cnn_stride = stride


