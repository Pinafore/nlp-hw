import argparse
import os
import json
from collections import Counter, defaultdict
from typing import Sequence, Dict

import numpy
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You may modify this code, but you shouldn't need to

        self.x = x
        self.y = y
        self.k = k

    def majority(self, item_indices: Sequence[int]) -> str:
        """Given the indices of training examples, return the majority label.
        If there's a tie, return the one that is lexicographically
        first (as determined by python sorted function).

        :param item_indices: The indices of the k nearest neighbors
        (helpfully, this is what's returned by the kneighbors
        function.
        """
        assert len(item_indices) == self.k, "Did not get k inputs"

        # Finish this function to return the most common y value for
        # these indices

        return None

    def classify(self, example: numpy.ndarray) -> str:
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """
        # Finish this function to find the k closest points, query the
        # majority function, and return the value.

        return ""

    def confusion_matrix(self, test_x: Sequence[str], test_y: Sequence[str]) -> Dict[str, Dict[str, int]]:
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        return d

    @staticmethod
    def accuracy(confusion_matrix: Dict[str, Dict[str, int]]) -> float:
        """Given a confusion matrix, compute the accuracy of the underlying
        classifier.

        """

        # Hint: this should give you clues as to how the confusion
        # matrix should be structured.

        total = 0 
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        return float(correct) / float(total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument("--root_dir", help="QB Dataset for training",
                        type=str, default='../',
                        required=False)
    parser.add_argument("--train_dataset", help="QB Dataset for training",
                        type=str, default='qanta.train.json',
                        required=False)
    parser.add_argument("--test_dataset", help="QB Dataset for test",
                        type=str, default='qanta.dev.json',
                        required=False)
    parser.add_argument("--min_df", help="How many documents must a word appear in to be feature",
                        type=int, default=2)
    parser.add_argument("--max_df", help="How many docs can words appear in and still be feature",
                        type=float, default=0.9)
    parser.add_argument("--limit", help="Number of training documents",
                        type=int, default=-1, required=False)
    parser.add_argument("--max_ngram", help="Max ngram length", type=int, default=3)
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    args = parser.parse_args()

    # You should not have to modify any of this code
    with open(os.path.join(args.root_dir, args.train_dataset)) as infile:
        data = json.load(infile)["questions"]
        if args.limit > 0:
            data = data[:args.limit]
    vectorizer = TfidfVectorizer(ngram_range=(1, args.max_ngram), min_df=args.min_df, max_df=args.max_df).fit(x["text"] for x in data)
    train_x = vectorizer.transform(x["text"] for x in data)
    train_y = list(x["page"] for x in data)

    print(type(train_x))

    knn = Knearest(train_x, train_y, args.k)
    print("Done loading data")

    with open(os.path.join(args.root_dir, args.test_dataset)) as infile:
        test = json.load(infile)["questions"][:100]

    test_x = vectorizer.transform(x["text"] for x in test)
    test_y = list(x["page"] for x in test)
    answers = [x[0] for x in Counter(test_y).most_common(5)]

    confusion = knn.confusion_matrix(test_x, test_y)
    guesses = set()
    for ii in answers:
        for jj in confusion[ii]:
            guesses.add(jj)

    print("\t" + "\t".join(str(x) for x in answers))
    print("".join(["-"] * 90))
    for ii in guesses:
        print("%30s:\t" % ii + "\t".join(str(confusion[x].get(ii, 0))
                                       for x in answers))
    print("Accuracy: %f" % knn.accuracy(confusion))
