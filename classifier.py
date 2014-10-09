import json
from math import log
import codecs
import sys
import random

class Model(object):
    
    JUNK = "__junk__"
    
    def __init__(self):
        self.unsmoothed = {}
        self.smoothed = {}
    
    def increment_given(self, item, given):
        """Increment the counts in the item's given table.
        item - The class, word, or otherwise that gets incremented.
        given - The tuple of given information.
        """
        self.unsmoothed.setdefault(given, [0, {}])
        temp = self.unsmoothed[given]
        temp[0] += 1
        temp[1][item] = temp[1].setdefault(item, 0) + 1
    
    def log(self, item, given):
        if given not in self.unsmoothed:
            return 0
        else:
            temp = self.unsmoothed[given]
            if item not in temp[1]:
                return 0
            return log(temp[1][item]) - log(temp[0])
    
    def check_sum_to_one(self):
        """Return True if the conditional probability tables sum to one each sum to one."""
        for given, count_table in self.unsmoothed.iteritems():
            count = count_table[0]
            items = count_table[1]
            if count == 0:
                return False
            else:
                total = 0
                for item, item_count in items.iteritems():
                    total += item_count
                if count != total:
                    return False
        return True
    
    def get_table_iterator(self, given):
        if given not in self.unsmoothed:
            return []
        else:
            return iter(self.unsmoothed[given][1])
    
    def get_number_of_tables(self):
        return len(self.unsmoothed)
    
    def get_average_table_length(self):
        total = len(self.unsmoothed)
        total_entries = 0
        for i, j in self.unsmoothed.iteritems():
            total_entries += len(j[1])
        return float(total_entries)/float(total)
    
    def print_model(self):
        print self.classes
        print self.words
    
    def to_json(self):
        return json.dumps(self.unsmoothed)
    
    def from_json(self, text):
        self.unsmoothed = json.loads(text)

class Classifier(object):
    
    START = "__start__"
    
    def __init__(self):
        self.model = Model()
    
    def train(self, class_name, token_list):
        """Train the model.
        class_name - The name of the class the token sequence identifies.
        token_iterator - A token iterator that returns tokens in the sequence they appear in the document.
        """
        self.model.increment_given(class_name, '')
        prev_prev_token = self.START
        prev_token = self.START
        for token in token_list:
            self.model.increment_given(token, str((class_name)))
            self.model.increment_given(token, str((class_name, prev_token)))
            self.model.increment_given(token, str((class_name, prev_prev_token, prev_token)))
            prev_prev_token = prev_token
            prev_token = token
        
    def print_model(self):
        self.model.print_model()
    
    def save_model(self, file_name):
        with codecs.open(file_name, 'w', 'utf-8') as f_out:
            f_out.write(self.model.to_json())
    
    def load_model(self, file_name):
        with codecs.open(file_name, 'r', 'utf-8') as f_in:
            self.model.from_json(f_in.read())
    
    def check(self):
        """Check that the model is not broken."""
        return self.model.check_sum_to_one()
    
    def print_stats(self):
        """Print out basic statistics."""
        print "Number of tables:", self.model.get_number_of_tables()
        print "Average table length:", self.model.get_average_table_length()
    
    def classify_random(self, token_list):
        """Use reservoir sampling to classify the token list with a seed of zero."""
        seed = 0
        random.seed(0)
        reservoir = []
        res_size = 1
        length = len(token_list)
        for i, class_name in enumerate(self.model.get_table_iterator('')):
            if i < res_size:
                reservoir.append(class_name)
            else:
                r = random.randint(0, i)
                if r < res_size:
                    reservoir[r] = class_name
        return reservoir[0]
    
    def classify(self, token_list):
        """Classify the given words with no smoothing. No previous tokens are taken into account.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            temp = self.model.log(class_name, '')
            for tok in token_list:
                temp += self.model.log(tok, str((class_name)))
            # Check for better class found.
            if temp >= max_log_prob:
                max_log_prob = temp
                max_class = class_name
        return max_class
    
    def classify_prev_token(self, token_list):
        """Classify the given words with no smoothing and one previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            temp = self.model.log(class_name, '')
            prev_token = self.START
            for token in token_list:
                temp += self.model.log(token, str((class_name, prev_token)))
                prev_token = token
            # Check for better class found.
            if temp >= max_log_prob:
                max_log_prob = temp
                max_class = class_name
        return max_class
    
    def classify_prev_prev_token(self, token_list):
        """Classify the given words with no smoothing and one previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            temp = self.model.log(class_name, '')
            prev_prev_token = self.START
            prev_token = self.START
            for token in token_list:
                temp += self.model.log(token, str((class_name, prev_prev_token, prev_token)))
                prev_prev_token = prev_token
                prev_token = token
            # Check for better class found.
            if temp >= max_log_prob:
                max_log_prob = temp
                max_class = class_name
        return max_class
