
import os
import re
import codecs
from classifier import Classifier

TOKEN_REGEX = r"."
TOKEN_PATTERN = re.compile(TOKEN_REGEX)

def token_iterator(line, pattern):
    for m in pattern.finditer(line):
        yield m.group().lower()

if __name__ == "__main__":
    # Train on proper nouns dataset
    c = Classifier()
    
    train_file = "assignment2/pnp-train.txt"
    #~ validate_file = "assignment2/pnp-validate.txt"
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
    assert c.check()
    print "Sanity check passed."
    c.print_stats()
    c.save_model("char_level_classifier.json")
