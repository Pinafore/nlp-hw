from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
import os

from typing import Union, Dict
import math
import logging

from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'

import os

from nltk.tokenize import sent_tokenize
from guesser import print_guess, Guesser

class DummyVectorizer:
    """
    A dumb vectorizer that only creates a random matrix instead of something real.
    """
    def __init__(self, width=50):
        self.width = width
    
    def transform(self, questions):
        import numpy as np
        return np.random.rand(len(questions), self.width)

class TfidfGuesser(Guesser):
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self, filename):
        """
        Initializes data structures that will be useful later.
        """
        # You'll need add the vectorizer here and replace this fake vectorizer
        self.tfidf_vectorizer = DummyVectorizer()
        self.tfidf = None 
        self.questions = None
        self.answers = None
        self.filename = filename

    def train(self, training_data, answer_field, split_by_sentence, min_length, max_length):
        """
        Train the guesser from the data
        """
        
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length, max_length)

        self.tfidf = self.tfidf_vectorizer.transform(self.questions)

    def save(self):
        """
        Save the parameters to disk
        """
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open("%s.tfidf.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf, f)

        with open("%s.questions.pkl" % path, 'wb') as f:
            pickle.dump(self.questions, f)

        with open("%s.answers.pkl" % path, 'wb') as f:
            pickle.dump(self.answers, f)

    def __call__(self, question, max_n_guesses=4):
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        question -- Raw text of the question
        max_n_guesses -- How many top guesses to return
        """
        top_questions = []
        top_answers = []
        top_sim = []

        # Compute the cosine similarity
        question_tfidf = self.tfidf_vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        cos = cosine_similarities[0]
        indices = cos.argsort()[::-1]
        guesses = []
        for i in range(max_n_guesses):
            # The line below is wrong but lets the code run for the homework.
            # Remove it or fix it!
            idx = i
            guess =  {"question": self.questions[idx], "guess": self.answers[idx],
                      "confidence": cos[idx]}
            guesses.append(guess)
        return guesses

    def load(self):
        """
        Load the tf-idf guesser from a file
        """
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open("%s.tfidf.pkl" % path, 'rb') as f:
            self.tfidf = pickle.load(f)
        
        with open("%s.questions.pkl" % path, 'rb') as f:
            self.questions = pickle.load(f)

        with open("%s.answers.pkl" % path, 'rb') as f:
            self.answers = pickle.load(f)


if __name__ == "__main__":
    
    all_qa = ["QANTA"]
    for QA_data in all_qa:
        os.makedirs(os.path.join(SAVE_DIR, QA_data), exist_ok = True)
        

        print("Loading %s" % guesstrain)
        with open(guesstrain, "r") as f:
            train = json.load(f)

        tfidf_guesser = TfidfGuesser()
        tfidf_guesser.train(train)
        tfidf_guesser.save(QA_data)    
        print("Loading %s" % guesstest)
        with open(guesstest, "r") as f:
            test = json.load(f)
   
        for qn in test:
            question = qn["question"]
            print (question)
            guesses = tfidf_guesser(question = question, max_n_guesses = 4)
            for ii in guesses:
                print("\t" + print_guess(ii))

   
    
