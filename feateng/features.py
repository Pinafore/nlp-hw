from collections import Counter
from math import log

class Feature:
    def __init__(self, name):
        self.name = name

class LengthFeature(Feature):
    def __call__(self, question, run, guess):
        yield ("char", log(1 + len(run)))
        yield ("guess", log(1 + len(guess)))
        yield ("word", log(1 + len(run.split())))




