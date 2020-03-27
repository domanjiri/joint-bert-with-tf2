from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl import app

import config
import preprocess
import model

def main(argv):
    del argv

    d_factory = preprocess.ProcessFactory(
        sentences=config.sentences_file,
        intents=config.intents_file,
        slots=config.slots_file,
        split=config.validation_set_ratio)
    data = d_factory.get_data()
    logging.info('after preprocess')

    joint_model = model.CategoricalBert(
        train=data['train'],
        validation=data['validation'],
        intents_num=d_factory.get_intents_num(),
        slots_num=d_factory.get_slots_num())

    joint_model.fit()



if __name__ == '__main__':
    app.run(main)

