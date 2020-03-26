from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
from absl import logging
import numpy as np

import tensorflow as tf
from bert import bert_tokenization

import config

class Process(object):

    def __init__(self, sentences, intents):
        self._dataset = {'sentence': sentences, 'intent': intents}
        self._tokenizer = bert_tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                                          do_lower_case=True)
        self._vectorize()

    def get_tokens(self) -> np.ndarray:
        return self._tokens

    def get_intents(self):
        return self._intents

    def _vectorize(self):

        def prepare_fn(data):
            data = ['[CLS]{}[SEP]'.format(x) for x in data.as_numpy_iterator()]
            return np.array([self._tokenizer.convert_tokens_to_ids(
                                self._tokenizer.tokenize(sentence))
                                    for sentence in data])

        def padding_fn(data, max_len=50):
            return tf.keras.preprocessing.sequence.pad_sequences(
                data,
                maxlen=max_len,
                truncating='post',
                padding='post')

        self._tokens = padding_fn(data=prepare_fn(self._dataset['sentence']),
                                  max_len=config.tokens_max_len)
        
        logging.info('sentences have processed')
        
        # make fixed lenght one_hot for each intent
        intents_set = set(x for x in self._dataset['intent'].as_numpy_iterator())
        categorical_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='intent',
            vocabulary_list=intents_set)
        self._intents = categorical_c #tf.feature_column.indicator_column(categorical_c)

        logging.info('intents prepared as one_hot tensors')

class ProcessFactory(object):

    def __init__(self,
                 sentences : str = '',
                 intents : str = '',
                 split : float = .2):
        assert 0 < split < 1, "split number must be between zero and one"

        self._split_size = split
        self._file_path = {
            'sentence' : sentences,
            'intent' : intents,
        }
        # load data from file to memory
        self._load_data()

    def get_data(self) -> Tuple[Process, Process]:
        test, train = self._split()
        return Process(*train), Process(*test)

    def get_intents_num(self) -> int:
        intents = set(x for x in self._dataset['intent'].as_numpy_iterator())
        return len(intents)

    def _load_data(self):
        self._dataset = {key : tf.data.TextLineDataset(value) 
                            for key, value in self._file_path.items()}
        
        # count data entities
        def count_fn(data):
            return len(list(data)) # TODO(Ebi): is there a way better than iteration? 
        entities = set(count_fn(x) for x in list(self._dataset.values()))
        
        # check for any corruption in files
        assert len(entities) == 1, "all files should have the same number of lines"

        self._entities_num = entities.pop()

        logging.info('file loaded into memory')

    def _split(self):
        test_part = int(self._entities_num * self._split_size)
        return (
        (self._dataset['sentence'].take(test_part), self._dataset['intent'].take(test_part)), #test
        (self._dataset['sentence'].skip(test_part), self._dataset['intent'].skip(test_part))) # train

