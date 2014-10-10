import json
from math import log
import codecs
import sys
import random
import copy
from ast import literal_eval # Used to parse tuples.

class Model(object):
    
    def __init__(self):
        self.internal_model = {}
    
    def __contains__(self, given):
        return str(given) in self.internal_model
    
    def iteritems(self):
        def f():
            for k, v in self.internal_model.iteritems():
                yield k, v[1]
        return f()
    
    def add_given(self, item, given, increment=1):
        """Increment the counts in the item's given table.
        item - The class, word, or otherwise that gets incremented.
        given - The tuple of given information.
        """
        given_key = str(given)
        self.internal_model.setdefault(given_key, [0, {}])
        temp = self.internal_model[given_key]
        temp[0] += increment
        temp[1][item] = temp[1].setdefault(item, 0) + increment
    
    def get_table(self, given):
        given_key = str(given)
        return self.internal_model[given_key]
    
    def log(self, item, given):
        """Return the numerator and the denominator of the probability as a tuple of logs.
        e.g. log(num/denom) = log(num)-log(denom) => return log(num), log(denom)
        If item and/or given are not found return 0, 0.
        The mods are used to modify the counts for smoothing.
        The slack_var is used if the original item is not found.
        """
        given_key = str(given)
        if given_key not in self.internal_model:
            return 0, 0
        else:
            temp = self.internal_model[given_key]
            if item not in temp[1]:
                return 0, 0
            else:
                return log(temp[1][item]), log(temp[0])
    
    def smoothed_log(self, item, given, numerator_mod=0, denominator_mod=0, slack_var=None, use_mods_for_junk_given=False):
        """Return the numerator and the denominator of the probability as a tuple of logs.
        e.g. log(num/denom) = log(num)-log(denom) => return log(num), log(denom)
        If item and/or given are not found return 0, 0.
        The mods are used to modify the counts for smoothing.
        The slack_var is used if the original item is not found.
        """
        given_key = str(given)
        if given_key not in self.internal_model:
            if use_mods_for_junk_given:
                return log(numerator_mod), log(denominator_mod)
            else:
                return 0, 0
        else:
            temp = self.internal_model[given_key]
            if item not in temp[1]:
                if slack_var not in temp[1]:
                    return 0, 0
                else:
                    return log(temp[1][slack_var] + numerator_mod), log(temp[0] + denominator_mod)
            else:
                return log(temp[1][item] + numerator_mod), log(temp[0] + denominator_mod)
    
    def joint_log(self, item, given, model):
        """Same as log only takes into account another model's counts."""
        count, total_count = self.get_given_counts(item, given)
        count2, total_count2 = model.get_given_counts(item, given)
        count += count2
        total_count += total_count2
        if count == 0 or total_count == 0:
            return 0
        else:
            return log(count), log(total_count)
    
    def get_given_counts(self, item, given):
        """Return a tuple of the counts of the item in the 
        given context and the total counts of the given context.
        """
        given_key = str(given)
        if given_key not in self.internal_model:
            return 0, 0
        else:
            table = self.internal_model[given_key]
            if item not in table[1]:
                return 0, table[0]
            else:
                return table[1][item], table[0]
    
    def check_sum_to_one(self):
        """Return True if the conditional probability tables sum to one each sum to one."""
        for given, count_table in self.internal_model.iteritems():
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
    
    def mimic(self, model):
        """Take and deep copy the contents of the given model."""
        self.internal_model = copy.deepcopy(model.internal_model)
    
    def get_table_iterator(self, given):
        """Return an iterator over the given table's keys."""
        if given not in self.internal_model:
            return []
        else:
            return iter(self.internal_model[given][1])
    
    def get_number_of_tables(self):
        return len(self.internal_model)
    
    def get_average_table_length(self):
        total = len(self.internal_model)
        total_entries = 0
        for i, j in self.internal_model.iteritems():
            total_entries += len(j[1])
        return float(total_entries)/float(total)
    
    def print_model(self):
        print self.classes
        print self.words

class Classifier(object):
    
    START = "__start__"
    JUNK = "__junk__"
    
    def __init__(self):
        self.classes = {}
        
        self.model = Model()
        self.smoothed_model = Model()
        self.types = {}
        
        self.semi_supervised_model = Model()
        self.semi_supervised_types = {}
    
    def _train_model(self, model, class_name, token_list, start_symbol):
        """Internal use."""
        model.add_given(class_name, '')
        prev_prev_token = start_symbol
        prev_token = start_symbol
        for token in token_list:
            model.add_given(token, (class_name,))
            model.add_given(token, (class_name, prev_token))
            model.add_given(token, (class_name, prev_prev_token, prev_token))
            prev_prev_token = prev_token
            prev_token = token
            self.types[token] = True
    
    def train(self, class_name, token_list):
        """Train the model.
        class_name - The name of the class the token sequence identifies.
        token_iterator - A token iterator that returns tokens in the sequence they appear in the document.
        """
        self.classes[class_name] = True
        self._train_model(self.model, class_name, token_list, self.START)
    
    def signal_end_of_training(self):
        """Required to get the smoothed models to work properly."""
        self.smoothed_model.mimic(self.model)
        for given, table in self.smoothed_model.iteritems():
            if given != '':
                self.smoothed_model.add_given(self.JUNK, given, 0)
        self.types[self.JUNK] = True
        self.semi_supervised_model.mimic(self.smoothed_model)
        self.semi_supervised_types = copy.deepcopy(self.types)
    
    def unsupervised_training(self, batch):
        throw_out_percent = 0.5
        sub_batch_size = 200
        batch_count = 0
        while len(batch) > 0:
            batch_count += 1
            sub_batch = batch[:sub_batch_size]
            batch = batch[sub_batch_size:]
            print "Subbatch:", batch_count
            print "Subbatch Size:", len(sub_batch)
            sub_batch_threshold = len(sub_batch)*throw_out_percent
            while len(sub_batch) > sub_batch_threshold:
                max_class = 'No Class'
                max_class_log = -sys.float_info.max
                max_class_index = 0
                # Find largest class log probability
                for i, token_list in enumerate(sub_batch):
                    c, log_c = self.classify_prev_prev_token_plus_one_special(token_list)
                    if log_c > max_class_log:
                        max_class_log = log_c
                        max_class = c
                        max_class_index = i
                # Train on most probable
                self._train_model(self.semi_supervised_model, max_class, sub_batch[max_class_index], self.START)
                del sub_batch[max_class_index]
                        
    
    def print_model(self):
        self.model.print_model()
    
    def save_model(self, file_name):
        with codecs.open(file_name, 'w', 'utf-8') as f_out:
            temp = {}
            temp['unsmoothed'] = self.model.internal_model
            temp['smoothed'] = self.smoothed_model.internal_model
            temp['types'] = self.types
            temp['semi-supervised'] = self.semi_supervised_model.internal_model
            temp['semi-supervised-types'] = self.semi_supervised_types
            temp['classes'] = self.classes
            f_out.write(json.dumps(temp))
    
    def load_model(self, file_name):
        with codecs.open(file_name, 'r', 'utf-8') as f_in:
            temp = json.loads(f_in.read())
            self.model.internal_model = temp['unsmoothed']
            self.smoothed_model.internal_model = temp['smoothed']
            self.types = temp['types']
            self.semi_supervised_model.internal_model = temp['semi-supervised']
            self.semi_supervised_types = temp['semi-supervised-types']
            self.classes = temp['classes']
    
    def check_model(self):
        """Check that the model is not broken."""
        return self.model.check_sum_to_one() and self.smoothed_model.check_sum_to_one() and \
               self.semi_supervised_model.check_sum_to_one()
    
    def print_stats(self):
        """Print out basic statistics."""
        print "Number of tables:", self.model.get_number_of_tables()
        print "Average table length:", self.model.get_average_table_length()
    
    def classify_random(self, token_list, seed=42):
        """Use reservoir sampling to classify the token list with a seed of zero."""
        random.seed(seed)
        reservoir = []
        res_size = 1
        length = len(token_list)
        for i, class_name in enumerate(self.model.get_table_iterator('')):
            if i < res_size:
                reservoir.append(class_name,)
            else:
                r = random.randint(0, i)
                if r < res_size:
                    reservoir[r] = class_name
        return reservoir[0]
    
    def classify_greedy(self, token_list):
        """Return most frequently occuring document."""
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            total1, total2 = self.model.log(class_name, '')
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify(self, token_list):
        """Classify the given words with no smoothing. No previous tokens are taken into account.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            total1, total2 = self.model.log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.model.log(token, (class_name,))
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
        
    def classify_plus_one(self, token_list):
        """Same as classify, but with plus one smoothing."""
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.smoothed_model.get_table_iterator(''):
            total1, total2 = self.smoothed_model.smoothed_log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.smoothed_model.smoothed_log(token, (class_name,), 1, len(self.types), self.JUNK, True)
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
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
            prev_token = self.START
            total1, total2 = self.model.log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.model.log(token, (class_name, prev_token))
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_prev_token_plus_one(self, token_list):
        """Classify the given words with plus one smoothing and one previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.smoothed_model.get_table_iterator(''):
            prev_token = self.START
            total1, total2 = self.smoothed_model.smoothed_log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.smoothed_model.smoothed_log(token, (class_name, prev_token), 1, len(self.types), self.JUNK, True)
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_prev_prev_token(self, token_list):
        """Classify the given words with no smoothing and two previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            prev_prev_token = self.START
            prev_token = self.START
            total1, total2 = self.model.log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.model.log(token, (class_name, prev_prev_token, prev_token))
                prev_prev_token = prev_token
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_prev_prev_token_plus_one(self, token_list):
        """Classify the given words with plus one smoothing and two previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.smoothed_model.get_table_iterator(''):
            prev_prev_token = self.START
            prev_token = self.START
            total1, total2 = self.smoothed_model.smoothed_log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.smoothed_model.smoothed_log(token, (class_name, prev_prev_token, prev_token), 1, len(self.types), self.JUNK, True)
                prev_prev_token = prev_token
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
        
    def classify_add_hoc(self, token_list):
        """Classify the given words with plus one smoothing and two previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.smoothed_model.get_table_iterator(''):
            prev_prev_token = self.START
            prev_token = self.START
            total1, total2 = 0, 0
            for token in token_list:
                temp1, temp2 = self.smoothed_model.smoothed_log(token, (class_name, prev_prev_token, prev_token), 1, len(self.types), self.JUNK, True)
                prev_prev_token = prev_token
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_prev_prev_token_plus_one_special(self, token_list):
        """Classify the given words with plus one smoothing and two previous token.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.semi_supervised_model.get_table_iterator(''):
            prev_prev_token = self.START
            prev_token = self.START
            total1, total2 = self.semi_supervised_model.smoothed_log(class_name, '')
            for token in token_list:
                temp1, temp2 = self.semi_supervised_model.smoothed_log(token, (class_name, prev_prev_token, prev_token), 1, len(self.types), self.JUNK, True)
                prev_prev_token = prev_token
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class, max_log_prob
    
    def classify_assume_seen(self, token_list):
        """Classify the given words assuming that the words are part of the training data for each class.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            temp_model = Model()
            self._train_model(temp_model, class_name, token_list, self.START)
            
            total1, total2 = self.model.joint_log(class_name, '', temp_model)
            for token in token_list:
                temp1, temp2 = self.model.joint_log(token, (class_name,), temp_model)
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_assume_seen_prev(self, token_list):
        """Classify the given words assuming that the words are part of the training data for each class using prev word.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            temp_model = Model()
            self._train_model(temp_model, class_name, token_list, self.START)
            
            total1, total2 = self.model.joint_log(class_name, '', temp_model)
            prev_token = self.START
            for token in token_list:
                temp1, temp2 = self.model.joint_log(token, (class_name, prev_token), temp_model)
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_assume_seen_prev_prev(self, token_list):
        """Classify the given words assuming that the words are part of the training data for each class using two prev word.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.model.get_table_iterator(''):
            temp_model = Model()
            self._train_model(temp_model, class_name, token_list, self.START)
            
            total1, total2 = self.model.joint_log(class_name, '', temp_model)
            prev_prev_token = self.START
            prev_token = self.START
            for token in token_list:
                temp1, temp2 = self.model.joint_log(token, (class_name, prev_prev_token, prev_token), temp_model)
                prev_prev_token = prev_token
                prev_token = token
                total1 += temp1
                total2 += temp2
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
    
    def classify_semi_supervised(self, token_list):
        """Classify the given words using a semi supervised model,
        for each class using two prev word.
        token_list - A list of tokens used to classify the document.
        Return the most probable class name.
        """
        
        max_class = "No Class"
        max_log_prob = -sys.float_info.max
        for class_name in self.semi_supervised_model.get_table_iterator(''):
            total1, total2 = self.semi_supervised_model.smoothed_log(class_name, '')
            prev_prev_token = self.START
            prev_token = self.START
            for token in token_list:
                given = (class_name, prev_prev_token, prev_token)
                temp1, temp2 = self.semi_supervised_model.smoothed_log(token, given, 1, len(self.semi_supervised_types), self.JUNK, True)
                total1 += temp1
                total2 += temp2
                prev_prev_token = prev_token
                prev_token = token
            total = total1 - total2
            # Check for better class found.
            if total >= max_log_prob:
                max_log_prob = total
                max_class = class_name
        return max_class
        #~ 
        #~ max_class = "No Class"
        #~ max_log_prob = -sys.float_info.max
        #~ for class_name in self.smoothed_model.get_table_iterator(''):
            #~ prev_prev_token = self.START
            #~ prev_token = self.START
            #~ total1, total2 = self.smoothed_model.smoothed_log(class_name, '')
            #~ for token in token_list:
                #~ temp1, temp2 = self.smoothed_model.smoothed_log(token, (class_name, prev_prev_token, prev_token), 1, len(self.types), self.JUNK, True)
                #~ prev_prev_token = prev_token
                #~ prev_token = token
                #~ total1 += temp1
                #~ total2 += temp2
            #~ total = total1 - total2
            #~ # Check for better class found.
            #~ if total >= max_log_prob:
                #~ max_log_prob = total
                #~ max_class = class_name
        #~ return max_class
