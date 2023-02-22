from time import sleep
from guesser import Guesser
from string import ascii_lowercase
from math import floor
from collections import defaultdict

import json
from json import JSONDecodeError
from tqdm import tqdm
import logging



from baseconv import BaseConverter



from nltk.corpus import stopwords
from guesser import alphanum

class GprGuesser(Guesser):
    """
    Class that uses the OpenAI API to generate answers to questions with hints
    from our own guessers.  Cache the results because we're cheap and want
    reproducability.
    """

    def __init__(self, cache_filename="data/gpt3_cache.json", num_examples=2, num_shards=9999, shard_vocab=ascii_lowercase, shard_prefix_length=5):
        """

        @param num_examples: How many retrieval results to include in GPT prompt
        """
        self.retrievers = {}
        self.cache = {}
        self.num_queries = 0
        self.num_shards = num_shards
        self.num_examples = num_examples
        self.cache_filename = cache_filename
        self.stopwords = set(stopwords.words("english"))
        for ii in ["one", "man"]:
            self.stopwords.add(ii)

        self.num_shards = num_shards
        self.shard_vocab = shard_vocab
        self.shard_prefix_length = shard_prefix_length
        self.shard_converter = BaseConverter(shard_vocab)

    def clean_for_shard(self, query):
        words = alphanum.split(query.lower())
        nospace = "".join(x for x in words if not x in self.stopwords)
        clean = "".join(x for x in nospace if x in self.shard_vocab)
        
        if len(clean) < self.shard_prefix_length:
            clean += "".join([self.shard_vocab[0]] * (self.shard_prefix_length - len(clean)))
        
        return clean[:self.shard_prefix_length]

    def shard(self, string):
        """
        Shard the string based on prefix of the string.  
        """

        clean_version = self.clean_for_shard(string)            
        value = int(self.shard_converter.decode(clean_version))
        shard = floor((value / (len(self.shard_vocab) ** self.shard_prefix_length + 1)) * self.num_shards)

        assert shard >= 0
        assert shard < self.num_shards, "%i outside of range" % shard

        return shard

              


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
        
        
    def save(self, suffix=""):
        """
        Save the API results to a file to save money and time for the future
        """

        if self.num_queries > 0:
            logging.info("Made %i new queries, saving to %s" % (self.num_queries, self.cache_filename))
            shards = defaultdict(dict)
            for ii in self.cache:
                shard = self.shard(ii)
                shards[shard][ii] = self.cache[ii]

            for shard in tqdm(shards):
                filename = "%s%05i%s" % (self.cache_filename, shard, suffix)
                with open(filename, 'w') as outfile:
                    json_object = json.dumps(shards[shard], indent=2)
                    outfile.write(json_object)            
        



            
    def load(self):
        """
        Load the cache of search results from a file
        """
        from glob import glob
        self.cache = {}
        for ii in glob("%s*" % self.cache_filename):
            try:
                with open(ii, 'r') as infile:
                    json_object = json.load(infile)
            except (IOError, JSONDecodeError) as e:
                json_object = {}
                print("Failed to load cache")
            self.cache.update(json_object)                
            logging.info("Reading %09i entries from %s, cache size is now %09i" % (len(json_object), ii, len(self.cache)))

        logging.info("%i entries added to cache" % len(self.cache))
        return self.cache




    

if __name__ == "__main__":

    gg = GprGuesser(num_shards=5, cache_filename="data/gtp3_cache")
    gg.init_retriever("questions", "models/TfidfGuesser")
    gg.init_retriever("wiki", "models/WikiGuesser")    
    
    for qq in ["For ten points, name this city home of 22 Downing street and captial of the England",
               "Rodriguez-Rivera et al created one algorithm for doing this for Geodesic's Great Circle which is both non-moving and reduces fragmentation. Another method for doing this portions off half of the target at any one time and is known as Cheney's Algorithm. The generational type of this process relies on",
               "Two Quakers in this novel are named Peleg and Bildad."]:
        print(qq)
        print("----------------------------------")
        print(gg.prompt(qq))
        print(gg(qq))

    gg.save(suffix=".json")

    
