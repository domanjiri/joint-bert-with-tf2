from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow_hub as hub
import tensorflow as tf
import bert
from tensorflow.keras.models import Model

FullTokenizer = bert.bert_tokenization.FullTokenizer

