from time import sleep
from guesser import Guesser

import json
from tqdm import tqdm
import logging






class GprGuesser(Guesser):
    """
    Class that uses the OpenAI API to generate answers to questions with hints
    from our own guessers.  Cache the results because we're cheap and want
    reproducability.
    """

    def __init__(self, cache_filename="data/gpt3_cache.json", num_examples=2):
        """

        @param num_examples: How many retrieval results to include in GPT prompt
        """
        self.retrievers = {}
        self.cache = {}
        self.num_queries = 0
        self.num_examples = num_examples
        self.cache_filename = cache_filename



    def __call__(self, question, n_guesses=1):
        """
        Generate a guess, but grab from the cache first if it's available
        """
        
        if n_guesses > 1:
            logging.warn("GPR Guesser doesn't support multiple guesses")

        # Check the cache, return it from there if we have it
        if question not in self.cache:
            result = None
            while result is None:
                    result = self.predict(question)

            self.cache[question] = result
        return [self.cache[question]]
        
        
    def save(self):
        """
        Save the API results to a file to save money and time for the future
        """
        
        if self.num_queries > 0:
            logging.info("Made %i new queries, saving to %s" % (self.num_queries, self.cache_filename))
            with open(self.cache_filename, 'w') as outfile:
                json_object = json.dumps(self.cache, indent=2)
                outfile.write(json_object)
            
    def load(self):
        """
        Load the cache of search results from a file
        """
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

    
