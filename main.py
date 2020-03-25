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

    # load date from files and vectorization
    preprocess.Process(
        sentences = config.sentences_file,
        labels = config.labels_file,
        intents = config.intents_file
    )

    logging.info('after preprocess')

if __name__ == '__main__':
    app.run(main)
