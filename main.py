from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl import app

import config
from preprocess import ProcessFactory
from model import JointCategoricalBert

def main(argv):
    del argv

    data_factory = ProcessFactory(
        sentences=config.sentences_file,
        intents=config.intents_file,
        slots=config.slots_file,
        split=config.validation_set_ratio)
    data = data_factory.get_data()
    logging.info('after preprocess')

    model = JointCategoricalBert(
        train=data['train'],
        validation=data['validation'],
        intents_num=data_factory.get_intents_num(),
        slots_num=data_factory.get_slots_num())

    model.fit()


if __name__ == '__main__':
    app.run(main)

