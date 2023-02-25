# Author: Jordan Boyd-Graber
# 2013

# File to take guesses and decide if they're correct

import argparse
import string
import logging
import pickle

from sklearn.feature_extraction import DictVectorizer
from unidecode import unidecode
from tqdm import tqdm


from collections import Counter

from guesser import add_guesser_params
from features import LengthFeature
from params import add_buzzer_params, add_question_params, load_guesser, load_buzzer, load_questions, add_general_params, setup_logging

def normalize_answer(answer):
    """
    Remove superflous components to create a normalized form of an answer that
    can be more easily compared.
    """
    if answer is None:
        return ''
    reduced = unidecode(answer)
    reduced = reduced.replace("_", " ")
    if "(" in reduced:
        reduced = reduced.split("(")[0]
    reduced = "".join(x for x in reduced.lower() if x not in string.punctuation)
    reduced = reduced.strip()

    for bad_start in ["the ", "a ", "an "]:
        if reduced.startswith(bad_start):
            reduced = reduced[len(bad_start):]
    return reduced.strip()

def rough_compare(guess, page):
    """
    See if a guess is correct.  Not perfect, but better than direct string
    comparison.  Allows for slight variation.
    """
    # TODO: Also add the original answer line
    if page is None:
        return False
    
    guess = normalize_answer(guess)
    page = normalize_answer(page)

    if guess == '':
        return False
    
    if guess == page:
        return True
    elif page.find(guess) >= 0 and (len(page) - len(guess)) / len(page) > 0.5:
        return True
    else:
        return False
    
def runs(text, run_length):
    """
    Given a quiz bowl questions, generate runs---subsegments that simulate
    reading the question out loud.

    These are then fed into the rest of the system.

    """
    words = text.split()
    assert len(words) > 0
    current_word = 0
    last_run = 0

    for idx in range(run_length, len(text), run_length):
        current_run = text.find(" ", idx)
        if current_run > last_run and current_run < idx + run_length:
            yield text[:current_run]
            last_run = current_run

    yield text

def sentence_runs(sentences, run_length):
    """
    Generate runs, but do it per sentence (always stopping at sentence boundaries).
    """
    
    previous = ""
    for sentence in sentences:
        for run in runs(sentence, run_length):
            yield previous + run
        previous += sentence
        previous += "  "
    
class Buzzer:
    """
    Base class for any system that can decide if a guess is correct or not.
    """
    
    def __init__(self, filename, run_length):
        self.filename = filename
        self._training = []
        self._correct = []
        self._features = []
        self._metadata = []
        self._feature_generators = []
        self._guessers = {}

        self._run_length=run_length
        logging.info("Buzzer using run length %i" % self._run_length)
        
        self._finalized = False
        self._primary_guesser = None
        self._classifier = None
        self._featurizer = None

    def add_guesser(self, guesser_name, guesser, primary_guesser=False):
        """
        Add a guesser identified by guesser_name to the set of guessers.

        If it is designated as the primary_guesser, then its guess will be
        chosen in the case of a tie.

        """

        assert not self._finalized, "Trying to add guesser after finalized"
        assert guesser_name != "consensus"
        assert guesser_name is not None
        assert guesser_name not in self._guessers
        self._guessers[guesser_name] = guesser
        if primary_guesser:
            self._primary_guesser = guesser_name

    def add_feature(self, feature_extractor):
        """
        Add a feature that the buzzer will use to decide to trust a guess.
        """

        assert not self._finalized, "Trying to add feature after finalized"
        assert feature_extractor.name not in [x.name for x in self._feature_generators]
        assert feature_extractor.name not in self._guessers
        self._feature_generators.append(feature_extractor)
        logging.info("Adding feature %s" % feature_extractor.name)

    def featurize(self, question, run_text):
        """
        Turn a question's run into features.
        """
        
        features = {}
        guess = None
        
        for gg in self._guessers:
            result = self._guessers[gg](run_text)
            result = list(result)[0]
            if gg == self._primary_guesser:
                guess = result["guess"]
            
            # features["%s_guess" % gg] = result["guess"]
            features["%s_confidence" % gg] = result["confidence"]


        for ff in self._feature_generators:
            for feat, val in ff(question, run_text, guess):
                features["%s_%s" % (ff.name, feat)] = val

        return guess, features

    def finalize(self):
        """
        Set the guessers (will prevent future addition of features and guessers)
        """
        
        self._finalized = True
        if self._primary_guesser is None:
            self._primary_guesser = "consensus"
        
    def add_data(self, questions, limit=-1, answer_field="page"):
        """
        Add data and extract features from them.
        """
        
        self.finalize()
        
        num_questions = 0
        for qq in tqdm(questions):
            answer = qq[answer_field]
            text = qq["text"]
            # Delete these fields so you can't inadvertently cheat while
            # creating features.  However, we need the answer for the labels.
            del qq[answer_field]
            del qq["text"]
            
            for rr in runs(text, self._run_length):
                assert len(self._features) == len(self._correct)
                guess, features = self.featurize(qq, rr)
                self._features.append(features)
                self._metadata.append({"guess": guess, "answer": answer, "id": qq["qanta_id"], "text": rr})
                correct = rough_compare(guess, answer)

                self._correct.append(correct)
            num_questions += 1

            if "GprGuesser" in self._guessers and num_questions % 10 == 1:
                self._guessers["GprGuesser"].save()
                
            if limit > 0 and num_questions > limit:
                break

        if "GprGuesser" in self._guessers:
            self._guessers["GprGuesser"].save()
            
        return self._features
    
    def single_predict(self, run):
        """
        Make a prediction from a single example ... this us useful when the code
        is run in real-time.

        """
        
        features = [self.featurize(None, run)]

        X = self._featurizer.transform(features)

        return self._classifier.predict(X), features[0]
    
           
    def predict(self, questions, online=False):
        """
        Predict from a large set of questions whether you should buzz or not.
        """
        
        assert self._classifier, "Classifier not trained"
        assert self._featurizer, "Featurizer not defined"
        X = self._featurizer.transform(self._features)

        return self._classifier.predict(X), X, self._features, self._correct, self._metadata
    
    def load(self):
        """
        Load the buzzer state from disk
        """
        
        with open("%s.featurizer.pkl" % self.filename, 'rb') as infile:
            self._featurizer = pickle.load(infile)        
    
    def save(self):
        """
        Save the buzzer state to disck
        """
        
        for gg in self._guessers:
            self._guessers[gg].save()
        with open("%s.featurizer.pkl" % self.filename, 'wb') as outfile:
            pickle.dump(self._featurizer, outfile)  
    
    def train(self):
        """
        Learn classifier parameters from the data loaded into the buzzer.
        """
        
        assert len(self._features) == len(self._correct)        
        self._featurizer = DictVectorizer(sparse=True)
        X = self._featurizer.fit_transform(self._features)
        return X

if __name__ == "__main__":
    # Train a simple model on QB data, save it to a file
    import argparse
    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_buzzer_params(parser)
    add_question_params(parser)
    flags = parser.parse_args()
    setup_logging(flags)    

    guesser = load_guesser(flags)    
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions, flags.limit)

    buzzer.train()
    buzzer.save()

    print("Ran on %i questions of %i" % (flags.limit, len(questions)))
    
