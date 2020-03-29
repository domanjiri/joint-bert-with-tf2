from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
from transformers import TFBertModel

import config
from preprocess import Process


class CustomBertLayer(tf.keras.layers.Layer):
    """Custom layer to modify build and call methods of BERT if needed.
    """

    def __init__(self, **kwargs):
        super(CustomBertLayer, self).__init__(**kwargs)
        self._bert = self._load_bert()

    def _load_bert(self):
        model = TFBertModel.from_pretrained(config.bert_model_name)
        
        logging.info('BERT weights loaded')
        return model

    def build(self, input_shape):
        super(CustomBertLayer, self).build(input_shape)

    def call(self, inputs):
        result = self._bert(inputs=inputs)
        
        return result


class CustomModel(tf.keras.Model):
    """Definition of the model to modify with custom call method.

    Args:
        intents_num(int):
            Number of intents in the working dataset that used in softmax layer.
        slots_num(int):
            Number of slots labels in the working dataset that used in softmax layer.
    """

    def __init__(self,
                 intents_num : int,
                 slots_num : int):
        super().__init__(name="joint_intent_slot")
        self._bert_layer = CustomBertLayer()
        self._dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self._intent_classifier = tf.keras.layers.Dense(intents_num,
                                                        activation='softmax',
                                                        name='intent_classifier')
        self._slot_classifier = tf.keras.layers.Dense(slots_num,
                                                      activation='softmax',
                                                      name='slot_classifier')

    def call(self, inputs, training=False, **kwargs):
        sequence_output, pooled_output = self._bert_layer(inputs, **kwargs)

        sequence_output = self._dropout(sequence_output, training)
        slot_logits = self._slot_classifier(sequence_output)

        pooled_output = self._dropout(pooled_output, training)
        intent_logits = self._intent_classifier(pooled_output)

        return slot_logits, intent_logits


class JointCategoricalBert(object):
    """Wrapper to model functions. The Model compiles with hyper-parameters and
    will be ready for fit.

    Args:
        train(preprocess.Process):
            Holds the training part of samples.
        validation(preprocess.Process):
            Holds the validation part of samples.
        intents_num(int):
            Number of intents in the working dataset which will be used in softmax layer.
        slots_num(int):
            Number of slot lables in the working dataset which will be used in softmax layer.
    """

    def __init__(self,
                 train : Process,
                 validation : Process,
                 intents_num : int,
                 slots_num : int): 
        self._dataset = {'train': train, 'validation': validation}
        self._model = CustomModel(intents_num=intents_num, slots_num=slots_num)
        self._compile()

    def _compile(self):
        """Compile the model with hyper-parameters that defined in the config file.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        losses = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
        loss_weights = [config.loss_weights['slot'], config.loss_weights['intent']]
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        self._model.compile(optimizer=optimizer,
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=metrics)
        logging.info("model compiled")


    def fit(self):
        """Fit the compiled model to the dataset. Hyper-parameters such as number of
        epochs defined in the config file.
        """
        logging.info('before fit model')
        self._model.fit(
            self._dataset['train'].get_tokens(),
            (self._dataset['train'].get_slots(), self._dataset['train'].get_intents()),
            validation_data=(
                self._dataset['validation'].get_tokens(),
                (self._dataset['validation'].get_slots(),
                    self._dataset['validation'].get_intents())),
            epochs=config.epochs_num,
            batch_size=config.batch_size)

        return self._model

