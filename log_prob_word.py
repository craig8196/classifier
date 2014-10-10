import os
import os.path as path
import re
import codecs
import json
import sys
from classifier import Classifier
from collections import OrderedDict
from train_classifier import token_iterator, TOKEN_PATTERN, TEST_FILE, TRAINING_FILE_OUTPUT


if __name__ == "__main__":
    c = Classifier()
    c.load_model(TRAINING_FILE_OUTPUT)
    item = unicode(sys.argv[1], 'utf-8')
    tokens = []
    for token in token_iterator(item, TOKEN_PATTERN):
        tokens.append(token)
    print "Smooth Log Prob (two token):", c.classify_prev_prev_token_plus_one_special(tokens)
