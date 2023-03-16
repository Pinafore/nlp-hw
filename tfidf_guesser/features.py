# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log

class Feature:
    """Base feature class.  Needs to be instantiated in params.py and then called

    by buzzer.py

    The name of the feature needs to be set in the constructor.  When
    the feature is called, it needs to return a tuple with the name of
    the feature as the first field and the value of the feature as the
    second field.
    """
    
    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess):
        raise NotImplementedError("Subclasses of Feature must implement this function")

class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    
    def __call__(self, question, run, guess):
        # How many characters long is the question?
        yield ("char", log(1 + len(run)))
        
        # How many words long is the question?
        yield ("word", log(1 + len(run.split())))

        # How many characters long is the guess?
        yield ("guess", log(1 + len(guess)))





