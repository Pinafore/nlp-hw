# Jordan Boyd-Graber
# 2023
#
# Buzzer using Logistic Regression

import pickle
import random
from math import exp, log
from collections import defaultdict
import json
import logging

import numpy as np

from buzzer import Buzzer

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Note: Prevents overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    # You'll want to use this function, but don't modify it
    
    if abs(score) > threshold:
        score = threshold * np.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, json_line, vocab, use_bias=True):
        """
        Create a new example

        json_line -- The json object that contains the label ("label") and features as fields
        vocab -- The vocabulary to use as features (list)
        use_bias -- Include a bias feature (should be false with Pytorch)
        """

        # Use but don't modify this function
        
        self.nonzero = {}
        self.y = 1 if json_line["label"] else 0
        self.x = np.zeros(len(vocab))

        for feature in json_line:
            if feature in vocab:
                assert feature != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(feature)] += float(json_line[feature])
                self.nonzero[vocab.index(feature)] = feature
        # Initialize the bias feature
        if use_bias:
            self.x[0] = 1

class ToyLogisticBuzzer(Buzzer):
    """
    Logistic regression classifier to predict whether a buzz is correct or not.
    """

    def __init__(self, num_features, mu=0.0, learning_rate=0.025, lazy=False):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param learning_rate: A function that takes the iteration as an argument (the default is a constant value)
        """

        self._lazy = lazy
        self._dimension = num_features
        self._beta = np.zeros(num_features)
        self._mu = mu
        self._step = learning_rate
        self._last_update = np.zeros(num_features)

        logging.info("Creating regression over %i features" % num_features)
        
        assert self._mu >= 0, "Regularization parameter must be non-negative"

        # You *may* want to add additional data members here
        self._beta = np.zeros(num_features)

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy, which is returned as a tuple.

        examples -- The dataset to score
        """

        # You probably don't need to modify this code
        
        logprob = 0.0

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        
        for ii in examples:
            p = sigmoid(self._beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if p < 0.5 and ii.y < 0.5:
                tn += 1
            elif p < 0.5 and ii.y >= 0.5:
                fn += 1
            elif p >= 0.5 and ii.y >= 0.5:
                tp += 1
            else:
                assert p >= 0.5 and ii.y < 0.5, "Impossible happened"
                fp += 1

        total = len(examples)
        return {"logprob": logprob,
                "acc":     (tp + tn) / total,
                "prec":    tp / (tp + fp + 0.00001),
                "recall":  tp / (fn + tp + 0.00001)}

    def sg_update(self, train_example, iteration):
        """
        Compute a stochastic gradient update to improve the log likelihood and return the new feature weights.

        train_example -- The example to take the gradient with respect to
        """
        mu = self._mu
        step = self._step
        beta = self._beta
        






        return beta

    def finalize_lazy(self, iteration):
        """
        After going through all normal updates, apply regularization to
        all variables that need it.

        Only implement this function if you do the lazy extra credit.
        """

        beta = self._beta
        return beta

    
    def inspect(self, vocab, limit=10):
        """
        A function to find the top features.
        """

        top = [0]
        bottom = []

        for idx in list(top) + list(bottom):
            logging.info("Feat %35s %3i: %+0.5f" %
                         (vocab[idx], idx, self._beta[idx]))

        return top, bottom

    
    def train(self, train = None, test = None, vocab=None, passes=1):
        """

        """

        if len(vocab) != self._dimension:
            logging.warn("Mismatch: vocab size is %s, but dimension is %i" % \
                         (len(vocab), self._dimension))

        # You don't need to modify this code
        if not train:
            train = []
            vocab = self._feature_generators
            features = [feature.name for feature in self._feature_generators]
            assert len(self._features) == len(self._correct)
            for x, y in zip(self._features, self._correct):
                x["label"] = self._correct
            train.append(Example(x, features))
        else:
            assert vocab is not None, \
                "Vocab must be supplied if we don't generate"

        update_number = 0
        for pass_num in range(passes):
            for ii in train:
                self._beta = self.sg_update(ii, update_number)
                update_number += 1                

                if update_number % 100 == 1:
                    train_progress = lr.progress(train)
                    if test:
                        test_progress = lr.progress(test)
                    else:
                        test_progress = defaultdict(int)
                        test_progress['logprob'] = float("-inf")

                    # lr.inspect(vocab)
                    message = "Update %6i\t" % update_number
                    for fold, progress in [("Train", train_progress),
                                           ("Dev", test_progress)]:
                        for stat in progress:
                            message += "%s%s = %0.3f\t" % (fold, stat, progress[stat])
                    logging.info(message)

        self._beta = self.finalize_lazy(update_number)
            
        self.inspect(vocab)
            

    def save(self):
        Buzzer.save(self)
        with open("%s.model.pkl" % self.filename, 'wb') as outfile:
            pickle.dump(self._classifier, outfile)

    def load(self):
        Buzzer.load(self)
        with open("%s.model.pkl" % self.filename, 'rb') as infile:
            self._classifier = pickle.load(infile)


def read_dataset(filename, vocab, limit):
    """
    Reads in a text dataset with a given vocabulary

    filename -- json lines file of the dataset 
    """

    # You should not need to modify this function
    
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    dataset = []
    num_examples = 0
    with open(filename) as infile:
        for line in infile:
            num_examples += 1
            ex = Example(json.loads(line), vocab)
            dataset.append(ex)

            if limit > 0 and num_examples >= limit:
                break

    # Shuffle the data so that we don't have order effects
    random.shuffle(dataset)

    return dataset

if __name__ == "__main__":
    import argparse    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--vocab", help="Vocabulary of all features",
                           type=str, default="../data/small_guess.vocab")
    argparser.add_argument("--train", help="Training set",
                           type=str, default="../data/small_guess.buzztrain.jsonl", required=False)
    argparser.add_argument('--regularization', type=float, default=0.0)
    argparser.add_argument('--learning_rate', type=float, default=0.1)
    argparser.add_argument("--limit", type=int, default=-1)
    argparser.add_argument("--test", help="Test set",
                           type=str, default="../data/small_guess.buzzdev.jsonl", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    with open(args.vocab, 'r') as infile:
        vocab = [x.strip() for x in infile]
    print("Loaded %i items from vocab %s" % (len(vocab), args.vocab))
        
    train = read_dataset(args.train, vocab=vocab, limit=args.limit)
    test = read_dataset(args.test, vocab=vocab, limit=args.limit)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = ToyLogisticBuzzer(len(vocab),
                           mu=args.regularization,
                           learning_rate=args.learning_rate)

    # Iterations
    update_number = 0
    lr.train(train, test, vocab, args.passes)
