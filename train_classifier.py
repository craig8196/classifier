# train_classifier.py

import os
import re
import codecs
from classifier import Classifier

TRAINING_FILE = "assignment2/pnp-train.txt"
VALIDATE_FILE = "assignment2/pnp-validate.txt"
TEST_FILE = "assignment2/pnp-test.txt"

TRAINING_FILE_OUTPUT = 'trained.json'

TOKEN_REGEX = r"."
TOKEN_PATTERN = re.compile(TOKEN_REGEX)

def token_iterator(line, pattern):
    for m in pattern.finditer(line):
        yield m.group().lower()
#~ def token_iterator(line, pattern):
    #~ for m in pattern.finditer(line):
        #~ yield m.group()

if __name__ == "__main__":
    # Train on proper nouns dataset
    c = Classifier()
    
    # Train
    train_file = TRAINING_FILE
    validate_file = VALIDATE_FILE
    training_files = [train_file]
    for file_name in training_files:
        with codecs.open(file_name, 'r', 'utf-8') as f:
            for line in f:
                class_name, text = line.split('\t', 1)
                text = text.strip()
                tokens = []
                for token in token_iterator(text, TOKEN_PATTERN):
                    tokens.append(token)
                c.train(class_name, tokens)
    c.signal_end_of_training()
    
    # Unsupervised learning
    # Note that optimally this would be done in the test file
    # but we do it here so the expensive process can be done once and then
    # saved to be used in testing.
    batch = [] # Unknown data.
    with codecs.open(TEST_FILE, 'r', 'utf-8') as f:
        for line in f:
            class_name, text = line.split('\t', 1)
            text = text.strip()
            tokens = []
            for token in token_iterator(text, TOKEN_PATTERN):
                tokens.append(token)
            batch.append(tokens)
    print "Number of unsupervised learning:", len(batch)
    c.unsupervised_training(batch)
    
    assert c.check_model()
    print "Sanity check passed."
    c.print_stats()
    c.save_model(TRAINING_FILE_OUTPUT)
