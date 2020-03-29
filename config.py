# tensorflow eager mode. set True for debugging
tf_eager_execution = False
# BERTs configs
bert_model_name = "bert-base-cased"
bert_model_dir = 'data/bert/wwm_uncased_L-24_H-1024_A-16'

# paths of dataset files on disc
sentences_file = 'data/sentences'
intents_file = 'data/intents'
slots_file = 'data/labels'

# data configs 
validation_set_ratio = 0.2

# compile and fit
tokens_max_len = 50
learning_rate = 5e-5
dropout_rate = 0.1
loss_weights = {'slot': 3., 'intent': 1.0}
epochs_num = 5
batch_size = 32

