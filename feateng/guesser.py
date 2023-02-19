# Jordan Boyd-Graber
# 2023

# Base class for our guessers

import os
import json

from nltk.tokenize import sent_tokenize

from params import load_guesser, load_questions, add_guesser_params, add_general_params, add_question_params

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
        None

    def load(self):
        """
        Does nothing here, but can be overridden in downstream classes to load
        state from a file.
        """
        None
        
    def train(self, training_data, answer_field, split_by_sentence, min_length,
              max_length, remove_missing_pages=True):
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """
        from collections import defaultdict
        from tqdm import tqdm
        
        answers_to_questions = defaultdict(set)
        if split_by_sentence:
            for qq in tqdm(training_data):
                for ss in sent_tokenize(qq["text"]):
                    if len(ss) > min_length and len(ss) < max_length:
                        answers_to_questions[qq[answer_field]].add(ss)
        else:
            for qq in tqdm(training_data):
                answers_to_questions[qq[answer_field]].add(x["text"])

        self.answers = []
        self.questions = []
        for answer in tqdm(answers_to_questions):
            if remove_missing_pages and answer is None or answer.strip() == '':
                continue
            for question in answers_to_questions[answer]:
                self.answers.append(answer)
                self.questions.append(question)
        
    def __call__(self, question, n_guesses=1):
        """
        Generate a guess from a question.
        """
        return [{"guess": self._default_guess, "confidence": 1.0}]


if __name__ == "__main__":
    # Train a tf-idf guesser and save it to a file
    import argparse
    parser = argparse.ArgumentParser()
    add_guesser_params(parser)
    add_general_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    
    guesser = load_guesser(flags)
    questions = load_questions(flags)
    # TODO(jbg): Change to use huggingface data, as declared in flags

    if flags.guesser_type == 'WikiGuesser':
        train_result = guesser.train(questions,
                                     flags.guesser_answer_field,
                                     flags.tfidf_split_sentence,
                                     flags.tfidf_min_length,
                                     flags.tfidf_max_length,
                                     flags.wiki_min_frequency)
        # The WikiGuesser has some results (text from asked about Wikipedia
        # pages) from saving and we want to cache them to a file
        guesser.save(train_result)        
    else:
        guesser.train(questions,
                      flags.guesser_answer_field,
                      flags.tfidf_split_sentence,
                      flags.tfidf_min_length,
                      flags.tfidf_max_length)
        guesser.save()
