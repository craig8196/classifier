import os
import os.path as path
import re
import codecs
import json
from classifier import Classifier
from collections import OrderedDict
from train_classifier import token_iterator, TOKEN_PATTERN, TEST_FILE, TRAINING_FILE_OUTPUT

def confusion_matrix_to_file(conf_matrix, file_name):
    axis = OrderedDict()
    for key in conf_matrix:
        axis[key[0]] = True
        axis[key[1]] = True
    for i, key in enumerate(axis):
        axis[key] = i
    matrix = [['' for i in xrange(len(axis) + 1)] for i in xrange(len(axis) + 1)]
    for i, key in enumerate(axis):
        matrix[0][i+1] = key
        matrix[i+1][0] = key
    for i, key in enumerate(axis):
        for j, key2 in enumerate(axis):
            i2 = i + 1
            j2 = j + 1
            comb_key = (key, key2)
            if comb_key in conf_matrix:
                matrix[i2][j2] = conf_matrix[comb_key]
            else:
                matrix[i2][j2] = 0
    with codecs.open(file_name, 'w', 'utf-8') as f:
        for line in matrix:
            for i, v in enumerate(line):
                line[i] = str(v)
            s = ', '.join(line)
            f.write(s+'\n')
            

def confused_docs_to_file(confused, file_name):
    with codecs.open(file_name, 'w', 'utf-8') as f:
        for c in confused:
            line = c[0]+' '+c[1]+' '+c[2]+'\n'
            f.write(line)
    
CONFUSED_MATRIX_POSTFIX = '_confusion_matrix.csv'
CONFUSED_POSTFIX = '_confused.txt'
CORRECT_POSTFIX = '_correct.txt'
CLASSIFIER_DATA_DIRECTORY = 'classifiers_data'

if __name__ == "__main__":
    c = Classifier()
    c.load_model(TRAINING_FILE_OUTPUT)
    test_file = TEST_FILE
    
    # Grab all test data from the file.
    test_data = []
    actual_class = []
    with codecs.open(test_file, 'r', 'utf-8') as f:
        for line in f:
            class_name, text = line.split('\t', 1)
            text = text.strip()
            token_list = []
            for token in token_iterator(text, TOKEN_PATTERN):
                token_list.append(token)
            test_data.append(token_list)
            actual_class.append(class_name)
    
    classifier_data_directory = CLASSIFIER_DATA_DIRECTORY
    if not path.exists(classifier_data_directory):
        os.mkdir(classifier_data_directory)
    
    
    print "Number of test documents:", len(test_data)
    # Use a suite of classifiers and gather statistics.
    classifiers = [
        ('random classifier', c.classify_random),
        ('greedy', c.classify_greedy),
        ('naive classifier no smoothing', c.classify),
        ('one word token classifier no smoothing', c.classify_prev_token),
        ('two word token classifier no smoothing', c.classify_prev_prev_token),
        ('naive classifier with smoothing', c.classify_plus_one),
        ('one word token classifier with smoothing', c.classify_prev_token_plus_one),
        ('two word token classifier with smoothing', c.classify_prev_prev_token_plus_one),
        ('assume_seen', c.classify_assume_seen),
        ('assume_seen one prev word token', c.classify_assume_seen_prev),
        ('assume_seen two prev word token', c.classify_assume_seen_prev_prev),
        ('semisupervised', c.classify_semi_supervised),
        ('adhoc', c.classify_add_hoc),
    ]
    for classifier_name, classifier in classifiers:
        confusion_matrix = {} # TODO save confusion_matrices for write-up
        confused_documents = [] # TODO save mis-classified text to file
        correct_documents = []
        success_count = 0
        for i, token_list in enumerate(test_data):
            predicted_class = classifier(token_list)
            
            key = (actual_class[i], predicted_class)
            if actual_class[i] == predicted_class:
                success_count += 1
                correct_documents.append([actual_class[i], predicted_class, ''.join(token_list)])
            else:
                confused_documents.append([actual_class[i], predicted_class, ''.join(token_list)])
            confusion_matrix[key] = confusion_matrix.setdefault(key, 0) + 1
        # Print output
        print "Percent Success ("+str(classifier_name)+"):", str(float(success_count)/float(len(test_data)))
        out_matrix_file = path.join(classifier_data_directory, classifier_name+CONFUSED_MATRIX_POSTFIX)
        out_confused_file = path.join(classifier_data_directory, classifier_name+CONFUSED_POSTFIX)
        out_correct_file = path.join(classifier_data_directory, classifier_name+CORRECT_POSTFIX)
        confusion_matrix_to_file(confusion_matrix, out_matrix_file)
        confused_docs_to_file(confused_documents, out_confused_file)
        confused_docs_to_file(correct_documents, out_correct_file)
        #~ print "Confused:", confusion_matrix
