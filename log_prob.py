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
    given_list = sys.argv[2:]
    for i, v in enumerate(given_list):
        given_list[i] = unicode(v, 'utf-8')
    given = tuple(given_list)
    if given[0] == '':
        given = ''
    print "Token:", item, "Given:", given
    n, d = c.model.log(item, given)
    print n-d, "(Raw Log)"
    n, d = c.smoothed_model.smoothed_log(item, given, 1, len(c.types), c.JUNK, True)
    print n-d, "(Smoothed Log)"
    n, d = c.semi_supervised_model.smoothed_log(item, given, 1, len(c.semi_supervised_types), c.JUNK, True)
    print n-d, "(Semi Supervised Log)"
    print c.smoothed_model.get_table(given)
