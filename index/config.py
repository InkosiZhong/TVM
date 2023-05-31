class IndexConfig:
    def __init__(self):
        self.cache_root = './index/cache'
        self.train_data = ''
        self.test_data = ''
        self.dec_type = None

        self.do_mining = True # Boolean that determines whether the mining step is skipped or not
        self.do_training = True # Boolean that determines whether the training/fine-tuning step of the embedding dnn is skipped or not
        self.train_reps = False # Boolean that determines whether the representative indexs are generated from train dataset
        self.do_remining = False # Boolean that determines whether the representative indexs are generated from train dataset
        self.do_infer = True # Boolean that allows you to either compute embeddings or load them from ./cache
        self.do_bucketting = True # Boolean that allows you to compute the buckets or load them from ./cache
        
        self.batch_size = 16 # general batch size for both the target and embedding dnn
        self.nb_train = 3000 # controls how many datapoints are labeled to perform the triplet training
        self.train_margin = 1.0 # controls the margin parameter of the triplet loss
        self.train_lr = 1e-4
        self.max_k = 5 # controls the k parameter described in the paper (for computing distance weighted means and votes)
        self.nb_buckets = 7000 # controls the number of buckets used to construct the index
        self.nb_training_its = 12000 # controls the number of datapoints are passed through the model during training
        self.seed = 1

    def eval(self):
        self.do_mining = False
        self.do_training = False
        self.do_remining = False
        self.do_infer = False
        self.do_bucketting = False
        return self