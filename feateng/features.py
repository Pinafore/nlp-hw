# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic

bert_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history):
        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history):
        # How many characters long is the question?
        yield ("char", (len(run) - 450) / 450)

        # How many words long is the question?
        yield ("word", (len(run.split()) - 75) / 75)

        ftp = 0

        # How many characters long is the guess?
        if guess is None or guess=="":
            yield ("guess", -1)
        else:
            yield ("guess", log(1 + len(guess)))
        
class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """
    def __call__(self, question, run, guess):
        yield ('true', len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """
    def __call__(self, question, run, guess):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))

class ContextualMatchFeature(Feature):
    """
    Feature that computes the semantic similarity between the question and guess.
    """
    def __init__(self, name):
        super().__init__(name)
        # Load a sentence transformer model to create embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Adjust model as needed

    def __call__(self, question, run, guess, guess_history):
        # Ensure guess is not empty
        if isinstance(question, dict):
            question = question.get("text", "")

        if isinstance(question, str) and guess and isinstance(guess, str):
            # Generate embeddings for question and guess
            question_embedding = self.model.encode(question, convert_to_tensor=True)
            guess_embedding = self.model.encode(guess, convert_to_tensor=True)

            # Calculate cosine similarity between question and guess
            similarity_score = util.pytorch_cos_sim(question_embedding, guess_embedding).item()

            # Yield the similarity score as a feature
            yield (self.name, similarity_score)
        else:
            # If guess is empty, yield a similarity score of zero
            yield (self.name, 0.0)

class FrequencyFeature(Feature):
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
            for ii in questions:
                self.counts[self.normalize(ii["page"])] += 1

    def __call__(self, question, run, guess, guess_history=None):
        yield ("guess", log(1 + self.counts[self.normalize(guess)]))

class CategoryFeature(Feature):                                          
    def __call__(self, question, run, guess, guess_history, other_guesses=None):              
        yield ("category", question["category"])                    
        yield ("year", log(question["year"]-1980))              
        yield ("subcategory", question["subcategory"])  
        yield ("tournament", question["tournament"])

class PreviousGuessFeature(Feature):                                                                    
    def __call__(self, question, run, guess, guess_history, other_guesses=None):                                         
        count = 0                                                                                       
        score = []                                                                                   
        for guesser in guess_history:                                                                   
            for time in guess_history[guesser]:                                                         
                # print(guess_history[guesser][time])                                                   
                count += sum(1 for x in guess_history[guesser][time] if x['guess'] == guess)             
                score += [x['confidence'] for x in guess_history[guesser][time] if x['guess'] == guess]  
        yield ("count", count)                                                                           
        yield ("max_score", max(score))                                                                  
        yield ("avg_score", mean(score))                                                                 

if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from params import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    add_guesser_params(parser)
    add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags)
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)

