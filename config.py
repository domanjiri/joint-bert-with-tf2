# tensorflow eager mode. set True for debugging
tf_eager_execution = False
# BERTs configs
bert_model_name = "bert-base-cased"

# paths of dataset files on disc
sentences_file = 'data/atis.train.query.csv'
intents_file = 'data/atis.train.intent.csv'
slots_file = 'data/atis.train.slots.csv'

# data configs 
validation_set_ratio = 0.05

# compile and fit
tokens_max_len = 50
learning_rate = 5e-5
dropout_rate = 0.1
loss_weights = {'slot': 3., 'intent': 1.0}
epochs_num = 5
batch_size = 32

