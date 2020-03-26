from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.models import Model
from bert import bert_tokenization

import config


class Process(object):

    def __init__(self,
                 sentences : str = '',
                 intents : str = ''):
        self._file_path = {
            'sentence' : sentences,
            'intent' : intents,
        }
        # load data from file to memory
        self.__load_data()

        self._tokenizer = bert_tokenization.FullTokenizer(vocab_file = config.vocab_file)
        self.__vectorize()

    def get_tokens(self):
        return self._tokens

    def get_intents(self):
        return self._intents

    def __load_data(self):
        self._dataset = {key : tf.data.TextLineDataset(value) 
                            for key, value in self._file_path.items()}
        
        # count data entities
        def count_fn(data):
            return len(list(data)) # TODO(Ebi): is there a way other than iteration? 
        entities = set(count_fn(x) for x in list(self._dataset.values()))
        
        # check for any corruption in files
        assert len(entities) == 1, "all files should have the same number of lines"

        logging.info('file loaded into memory')

    def __vectorize(self):

        def prepare_fn():
            _tokens = []
            data = [x.as_numpy_iterator() for x in self._dataset.values()]
            for sentence, intent in zip(*data): # TODO(Ebi): should I iterate just through sentences?
                tokens = self._tokenizer.tokenize(sentence)
                _tokens.append(self._tokenizer.convert_tokens_to_ids(tokens))
            return np.array(_tokens)

        def padding_fn(data, max_len=50):
            return tf.keras.preprocessing.sequence.pad_sequences(
                data,
                maxlen=max_len,
                truncating='post',
                padding='post')

        self._tokens = padding_fn(data=prepare_fn(), max_len=config.tokens_max_len)
        
        # make fixed lenght one_hot for each intent
        self._intents = tf.feature_column.categorical_column_with_vocabulary_list(
            key='intent',
            vocabulary_list=set(x for x in self._dataset['intent'].as_numpy_iterator()))

