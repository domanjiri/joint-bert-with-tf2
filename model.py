from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import math
import tensorflow_hub as hub
import tensorflow as tf
import bert
from tensorflow.keras.models import Model

import config
from preprocess import Process

class CategoricalBert(object):
    
    def __init__(self,
                 train : Process,
                 test : Process,
                 intents_num : int):
        self._num_fine_tune_layers = config.num_fine_tune_layers
        self._intents_num = intents_num
        self._dataset = {
            'sentence': train.get_tokens(),
            'intent': train.get_intents(),
            'test_sentence': test.get_tokens(),
            'test_intent': test.get_intents(),
        }
        
        self._bert_layer = self._load_bert_layer()
        self._model = self._definition()
        self._fit()

    def _load_bert_layer(self):
        bert_params = bert.params_from_pretrained_ckpt(config.bert_model_dir)
        bert_layer = bert.BertModelLayer.from_params(bert_params, name='bert')
        # we will not retrain bert model
        bert_layer.apply_adapter_freeze()

        logging.info('bert layer created')
        return bert_layer

    def _definition(self):
        max_seq_length = config.tokens_max_len

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32'),
            self._bert_layer,
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(self._intents_num, activation='softmax'),])

        model.build(input_shape=(None, max_seq_length))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=5e-5),
            loss_weights=[1.0],
            metrics=['accuracy'])

        model.summary()
        logging.info('sequential model created')
        return model

    def _fit(self):
        eval_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=config.bert_weights_file,
            save_weights_only=True,
            verbose=1)
        
        logging.info('before fit model')
        self._model.fit(
            self._dataset['sentence'],
            self._dataset['intent'],
            validation_data=(
                self._dataset['test_sentence'],
                self._dataset['test_intent']),
            epochs=5,
            verbose=1,
            batch_size=32,
            callbacks=[eval_callback])

