from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, Set
from absl import logging
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

import config


class Process(object):
    """Tokenizing data and making fixed-width vectors from each sample.

    Args:
        sentence(tf.data.Dataset):
            Raw samples that should be tokenized.
        intent(tf.data.Dataset):
            Single words intents. The order matches with samples
        slot:(tf.data.Dataset):
            Row lables separated by space. The order matches with samples
        intents_num(int):
            Number of total intents in working dataset
        slots_num(int):
            Number of total slots in working dataset
    """

    def __init__(self,
                 sentence,
                 intent,
                 slot,
                 intents_set,
                 slots_set):
        self._dataset = {'sentence': sentence, 'intent': intent, 'slot': slot}
        self._intents_set = intents_set
        self._slots_set = slots_set
        #self._intents_num = len(intents_set)
        self._tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self._vectorize()

    def get_tokens(self) -> np.ndarray:
        """Returns array of fixed-width tokenized samples.
        """
        return self._tokens

    def get_intents(self) -> np.ndarray:
        """Returns one-D array of intents id.
        """
        return self._intents

    def get_slots(self) -> np.ndarray:
        """Returns array of fixed width normalized slots labels.
        """
        return self._slots

    def _vectorize(self):
        max_len = config.tokens_max_len
        assert max_len > 0, "max length of token vector should be greater than zero"

        ids = np.empty((0, max_len), dtype=np.int32)
        masks = np.empty((0, max_len), dtype=np.int32)
        self._tokens = {'input_ids': ids, 'attention_masks': masks}

        def prepare_fn(sentence):
            id = self._tokenizer.encode(str(sentence),
                                        max_length=max_len)
            post_pad = max_len - len(id)
            mask = np.concatenate((np.ones(len(id), dtype=np.int32), np.zeros(post_pad, dtype=np.int32)))
            id = np.concatenate((id, np.zeros(post_pad, dtype=np.int32)))

            self._tokens['input_ids'] = np.concatenate(
                (self._tokens['input_ids'], [id]),
                axis=0)
            self._tokens['attention_masks'] = np.concatenate(
                (self._tokens['attention_masks'], [mask]),
                axis=0)

        any(prepare_fn(x) for x in self._dataset['sentence'].as_numpy_iterator())
        assert self._tokens['input_ids'].shape == self._tokens['attention_masks'].shape, "masks and ids did not match"
        logging.info('sentences have processed')

        intents_list = list(self._intents_set)
        #def one_hot_fn(seek):
        #    vector = np.zeros(self._intents_num, dtype=np.int32)
        #    np.put(vector, intents_list.index(seek), 1)
        #    return vector
        self._intents = np.array([intents_list.index(x)
                                    for x in self._dataset['intent'].as_numpy_iterator()])
        logging.info('intents prepared as one-hot vectors')

        # make fixed length multi-hot for slots
        slots_list = list(self._slots_set)
        def multi_hot_fn(seek):
            seek = seek.decode('utf-8').split(' ')
            ids = np.array([slots_list.index(x) for x in seek], dtype=np.int32)
            
            post_pad = max_len - len(ids)
            vector = np.concatenate((ids, np.zeros(post_pad, dtype=np.int32)))
            return vector
        self._slots = np.array([multi_hot_fn(x)
                                    for x in self._dataset['slot'].as_numpy_iterator()])
        logging.info('slots labels prepared as multi-hot vectors')


class ProcessFactory(object):
    """Load data from file and split samples in
    training and validation sets.
    
    Args:
        sentences(str):
            Path to the samples file
        intents(str):
            Path to the intent file.
            The order of intents must matches with samples.
        slots(str):
            Path to the slots labels file.
            The order of labels should matches with samples.
        split(float):
            The portion of data allocated to validation process.
            Split value must be grater than zero and less than one. This means
            the training must have a validation part.
    """

    def __init__(self,
                 sentences : str = '',
                 intents : str = '',
                 slots : str = '',
                 split : float = .2):
        assert 0 < split < 1, "split number must be between zero and one"

        self._split_size = split
        self._file_path = {
            'sentence': sentences,
            'intent': intents,
            'slot': slots,
        }
        # load data from file to memory
        self._load_data()

    def get_data(self) -> Tuple[Process, Process]:
        """Returns tuple of Process instances. The first element holds
        training and the second holds validation part of samples.
        """
        splitted = self._split_samples()
        return {'train':
                    Process(**splitted['train'],
                            intents_set=self.get_intents_set(),
                            slots_set=self.get_slots_set()),
                'validation':
                    Process(**splitted['validation'],
                            intents_set=self.get_intents_set(),
                            slots_set=self.get_slots_set())}

    def get_intents_num(self) -> int:
        """Returns number of intents of the working dataset.
        """
        return len(self.get_intents_set())

    def get_intents_set(self) -> Set:
        """Returns a Python set from all intents of the working dataset.
        """
        return set(x for x in self._dataset['intent'].as_numpy_iterator())

    def get_slots_num(self) -> int:
        """Returns number of slots labels of the working dateset.
        """
        return len(self.get_slots_set())

    def get_slots_set(self) -> Set:
        """Return a Python set from all slots labels of working dataset.
        """
        all_slots_label = []
        def extract_labels_fn(chain):
            any(all_slots_label.append(x) for x in chain.decode('utf-8').split(' '))

        any(extract_labels_fn(x) for x in self._dataset['slot'].as_numpy_iterator())

        return set(all_slots_label)

    def _load_data(self):
        """Load data from text files into the tf.data.Dataset objects.
        """
        self._dataset = {key : tf.data.TextLineDataset(value) 
                            for key, value in self._file_path.items()}
        
        def count_fn(data):
            return len(list(data)) 

        samples = set(count_fn(x) for x in list(self._dataset.values()))
        assert len(samples) == 1, "all files should have the same number of lines"

        self._samples_num = samples.pop()
        logging.info('file loaded into memory')

    def _split_samples(self):
        """Divides the samples to training and validation part.
        """
        validation_part = int(self._samples_num * self._split_size)
        validation = {key: value.take(validation_part) for key, value in self._dataset.items()}
        train = {key: value.skip(validation_part) for key, value in self._dataset.items()}

        logging.info("{} samples for training and {} for validation".format(
            self._samples_num - validation_part,
            validation_part))
        return {'validation': validation, 'train': train}

