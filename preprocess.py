from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
import tensorflow_text as text
import bert
from tensorflow.keras.models import Model

import config

FullTokenizer = bert.bert_tokenization.FullTokenizer

class Process(object):

    def __init__(self,
                 sentences : str = '',
                 labels : str = '',
                 intents : str = ''):
        self.sentences_path = sentences
        self.labels_path = labels
        self.intents_path = intents

        self.__load_data() # load data from file to memory

        self.__tokenize()

    def get_ids(self):
        ...

    def get_masks(self):
        ...

    def get_segments(self):
        ...

    def __load_data(self):
        self.s_dataset = tf.data.TextLineDataset(self.sentences_path)
        self.l_dataset = tf.data.TextLineDataset(self.labels_path)
        self.i_dataset = tf.data.TextLineDataset(self.intents_path)

        logging.info('file loaded into memory')

    def __padding(self, seq):
        ...

    def __tokenize(self):
        ...


