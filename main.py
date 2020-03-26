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

    dataset_factory = preprocess.ProcessFactory(
        sentences=config.sentences_file,
        intents=config.intents_file,
        split=config.train_test_split
    )
    train_set, test_set = dataset_factory.get_data()

    logging.info('after preprocess')

    joint_model = model.CategoricalBert(
        train=train_set,
        test=test_set,
        intents_num=dataset_factory.get_intents_num()
        )



if __name__ == '__main__':
    app.run(main)
