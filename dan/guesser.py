# Jordan Boyd-Graber
# 2023

# Base class for our guessers

import os
import re
import json
import logging

from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple

from nltk.tokenize import sent_tokenize, word_tokenize

alphanum = re.compile('[^a-zA-Z0-9]')

from params import load_guesser, load_questions, setup_logging
from params import add_general_params, add_guesser_params, add_general_params, add_question_params

kTOY_JSON = [{"text": "capital England", "page": "London"},
             {"text": "capital Russia", "page": "Moscow"},
             {"text": "currency England", "page": "Pound"},
             {"text": "currency Russia", "page": "Rouble"}]

def word_overlap(query, page):
    query_words = set(alphanum.split(query))
    page_words = set(alphanum.split(page))

    return len(query_words.intersection(page_words)) / len(query_words)


def print_guess(guess, max_char=20):
    """
    Utility function for printing out snippets (up to max_char) of top guesses.
    """
    
    standard = ["guess", "confidence", "question"]
    output = ""

    for ii in standard:
        if ii in guess:
            if isinstance(guess[ii], float):
                short = "%0.2f" % guess[ii]
            else:
                short = str(guess[ii])[:max_char]
            output += "%s:%s\t" % (ii, short)
            
    return output


class Guesser:
    """
    Base class for guessers.  If it itself is instantiated, it will only guess
    one thing (the default guess).  This is useful for unit testing.
    """
    
    def __init__(self, default_guess="Les Misérables (musical)"):
        self._default_guess = default_guess
        self.phrase_model = None
        None

    @staticmethod
    def split_examples(training_data, answer_field, split_by_sentence=True, min_length=-1, max_length=-1):
        from collections import defaultdict
        from tqdm import tqdm
        
        answers_to_questions = defaultdict(set)
        if split_by_sentence:
            for qq in tqdm(training_data):
                for ss in sent_tokenize(qq["text"]):
                    if (min_length < 0 or len(ss) > min_length) and (max_length < 0 or len(ss) < max_length):
                        answers_to_questions[qq[answer_field]].add(ss)
        else:
            for qq in tqdm(training_data):
                text = qq["text"]
                if (min_length < 0 or len(text) > min_length) and (max_length < 0 or len(text) < max_length):
                    answers_to_questions[qq[answer_field]].add(qq["text"])
        return answers_to_questions

    @staticmethod
    def filter_answers(questions_keyed_by_answers, remove_missing_pages=False, answer_lookup=None):
        """
        Remove missing answers or answers that aren't included in lookup.
        """
        
        from tqdm import tqdm        
        answers = []
        questions = []
        for answer in tqdm(questions_keyed_by_answers):
            if remove_missing_pages and answer is None or answer is not None and answer.strip() == '':
                continue
            elif answer_lookup is not None and answer not in answer_lookup:
                continue
            for question in questions_keyed_by_answers[answer]:
                answers.append(answer)
                questions.append(question)

        return questions, answers
        

    def train(self, training_data, answer_field, split_by_sentence, min_length=-1,
              max_length=-1, remove_missing_pages=True):
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        min_length -- ignore all text segments less than this length (-1 for no limit)
        max_length -- ingore all text segments longer than this length (-1 for no length)
        remove_missing_pages -- remove pages without an answer_field
        """

        answers_to_questions = self.split_examples(training_data, answer_field, split_by_sentence,
                                                   min_length, max_length)
        self.questions, self.answers = self.filter_answers(answers_to_questions)

        return answers_to_questions

    def find_phrases(self, questions : Iterable[str]):
        """
        Using the training question, find phrases that ofen appear together.

        Saves the resulting phrase detector to phrase_model, which can
        then be used to tokenize text using the phrase_tokenize
        function.
        """
        assert len(questions) > 0, "Cannot find phrases without questions"
        from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

        # TODO: it might be good to exclude punctuation here
        sentences = []
        for qq in self.questions:
            for ss in sent_tokenize(qq):
                sentences.append(word_tokenize(ss))

        self.phrase_model = Phrases(sentences, connector_words=ENGLISH_CONNECTOR_WORDS, min_count=30)

    def phrase_tokenize(self, question: str) -> Iterable[str]:
        """
        Given text (a question), tokenize the text and look for phrases.
        """
        assert self.phrase_model is not None
        # Todo: perhaps include additional normalization in this function (e.g., lemmatization)
        return self.phrase_model[word_tokenize(question)]
        

    def batch_guess(self, questions, n_guesses=1):
        """
        Given a list of questions, create a batch set of predictions.

        This should be overridden my more efficient implementations in subclasses.
        """
        from tqdm import tqdm
        guesses = []
        logging.info("Generating guesses for %i new question" % len(questions))
        for question in tqdm(questions):
            new_guesses = self(question, n_guesses)
            guesses.append(new_guesses)
        return guesses

    def save(self):
        path = self.model_filename
        if self.phrase_model is not None:
            filename = "%s.phrase.pkl" % path
            logging.info("Writing Guesser phrases to %s" % filename)
            self.phrase_model.save(filename)

    def load(self):
        path = self.model_filename        
        filename = "%s.phrase.pkl" % path
        try:
            from gensim.models.phrases import Phrases
            self.phrase_model = Phrases.load(filename)
        except FileNotFoundError:
            self.phrase_model = None
                
    def __call__(self, question, n_guesses=1):
        """
        Generate a guess set from a single question.
        """
        return [{"guess": self._default_guess, "confidence": 1.0}]


if __name__ == "__main__":
    # Train a guesser and save it to a file
    import argparse
    parser = argparse.ArgumentParser()
    add_general_params(parser)    
    add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)    
    guesser = load_guesser(flags)
    questions = load_questions(flags)
    # TODO(jbg): Change to use huggingface data, as declared in flags

    if flags.guesser_type == 'WikiGuesser':
        guesser.init_wiki(flags.wiki_zim_filename)        
        train_result = guesser.train(questions,
                                     flags.guesser_answer_field,
                                     flags.tfidf_split_sentence,
                                     flags.tfidf_min_length,
                                     flags.tfidf_max_length,
                                     flags.wiki_min_frequency)
        # The WikiGuesser has some results (text from asked about Wikipedia
        # pages) from saving and we want to cache them to a file
        guesser.save()
    else:
        guesser.train(questions,
                      flags.guesser_answer_field,
                      flags.tfidf_split_sentence,
                      flags.tfidf_min_length,
                      flags.tfidf_max_length)
        # DAN Guesser 
        if flags.guesser_type == "DanGuesser":
            dev_exs = load_questions(flags, secondary=True)
            guesser.set_eval_data(dev_exs)
            guesser.train_dan()
        guesser.save()
