from typing import List, Tuple
import nltk
import sklearn
import transformers
import numpy as np
import pandas as pd


class QuizBowlModel:

    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.

        Do NOT load your model or resources in the guess_and_buzz function, as it will
        increase latency severely. 
        """
        pass

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts list of question string, and returns a string 
        representing the guess and a corresponding boolean representing 
        whether or not to buzz. 

        If you are using a deep learning model, try to 
        use batched prediction instead of iterating using a for loop.
        """
        pass
