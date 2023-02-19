from time import sleep
from guesser import Guesser

import json
import tqdm
import logging


import openai

from openai.error import RateLimitError



class GprGuesser(Guesser):

    def __init__(self, cache_filename="data/gpt3_cache.json", num_examples=2):
        self.retrievers = {}
        self.cache = {}
        self.num_queries = 0
        self.num_examples = num_examples
        self.cache_filename = cache_filename



    def __call__(self, question, n_guesses=1):
        if n_guesses > 1:
            logging.warn("GPR Guesser doesn't support multiple guesses")
        
        if question not in self.cache:
            result = None
            while result is None:
                try:
                    result = self.predict(question)
                except RateLimitError:
                    logging.info("Rate limit error, waiting 10 seconds")
                    for _ in tqdm(10):
                        sleep(1)

            self.cache[question] = result
        return [self.cache[question]]
        
        
    def save(self):
        if self.num_queries > 0:
            logging.info("Made %i new queries, saving to %s" % (self.num_queries, self.cache_filename))
            with open(self.cache_filename, 'w') as outfile:
                json_object = json.dumps(self.cache, indent=2)
                outfile.write(json_object)
            
    def load(self):
        try:
            with open(self.cache_filename, 'r') as infile:
                json_object = json.load(infile)
            print("Loaded %i entries from cache" % len(json_object))
        except IOError:
            json_object = {}
            print("Failed to load cache")

        self.cache = json_object




    

if __name__ == "__main__":

    gg = GprGuesser()
    gg.init_retriever("questions", "models/TfidfGuesser")
    gg.init_retriever("wiki", "models/WikiGuesser")    
    
    for qq in ["For ten points, name this city home of 22 Downing street and captial of the England",
               "Rodriguez-Rivera et al created one algorithm for doing this for Geodesic's Great Circle which is both non-moving and reduces fragmentation. Another method for doing this portions off half of the target at any one time and is known as Cheney's Algorithm. The generational type of this process relies on",
               "Two Quakers in this novel are named Peleg and Bildad."]:
        print(qq)
        print("----------------------------------")
        print(gg.prompt(qq))
        print(gg(qq))

    gg.save()

    
