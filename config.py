bert_model_uri = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
bert_model_dir = 'data/bert/wwm_uncased_L-24_H-1024_A-16'
bert_weights_file = 'data/bert//weights.ckpt' 
vocab_file = "data/bert/wwm_uncased_L-24_H-1024_A-16/vocab.txt"
sentences_file = 'data/sentences'
labels_file = 'data/labels'
intents_file = 'data/intents'

model = {
    
}
train_test_split = 0.2
tokens_max_len = 50
num_fine_tune_layers = 12
