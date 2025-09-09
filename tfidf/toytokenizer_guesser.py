import argparse
import json
import logging
import pickle
from collections import defaultdict, Counter, deque, OrderedDict
from math import log
import os

from typing import Iterable, Tuple, Dict, Union

from tqdm import tqdm
import numpy as np
import numpy.typing as npt

from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk import FreqDist

from guesser import Guesser, kTOY_DATA

kUNK = "<UNK>"
kEND_STRING = "<ENDOFTEXT>"

def log10(x):
    return log(x) / log(10)

def lower(str):
    return str.lower()

class Vocab:
    """
    Class that tracks a mapping of words to integers.
    """

    
    def __init__(self, unknown:str=kUNK):
        """
        Create the Vocab object

        Args:
            unknown: The string that represents unknown tokens
        """
        self._id_to_word = {}
        self._word_to_id = {}
        self.final = False
        self._unk = unknown

        # Initialize the vocab with individual tokens
        for token in range(256):
            self.add(chr(token), token)

        self.add(unknown)

    def __contains__(self, candidate: Union[int, str]) -> bool:
        """
        Check to see if something is part of the mapping

        Args:
            candidate: The element we're checking to see if it's in the mapping
        """
        if isinstance(candidate, str):
            return candidate in self._word_to_id
        elif isinstance(candidate, int):
            return candidate in self._id_to_word
        else:
            return False
    
    def __iter__(self) -> Iterable[Tuple[str, int]]:
        """
        Returns:
            Provides an iterator over all elements in the mapping
        """
        for key in sorted(self._id_to_word.keys()):
            yield key, self._id_to_word[key] 

    def __len__(self):
        """
        Returns:
            How many items are in the vocab
        """
        return len(self._id_to_word)
            
    @staticmethod
    def string_from_bytes(self, bytestream: list[int], max_chars:int=1) -> str:
        """
        Given a bytetream, turn it into a string (if you can).

        Args:
            bytestream: Integers that may or may not represent a string
            max_chars: Do not return the string if it's longer than this many unicode characters
        Returns:
            UTF-8 representation of the bytestream if possible, None otherwise

        There's a more grounded version I did not use here:
        https://heycoach.in/blog/utf-8-validation-solution-in-python/
        """
        try:
            result = bytearray(bytestream).decode('utf-8')
        except UnicodeDecodeError:
            result = None

        if result and len(result) <= max_chars:
            return result
        else:
            return result

    def add(self, word: str, idx: int=-1) -> int:
        """
        Add a word to the vocab and return its index.

        Args:
            word: The word to add to the vocabulary
            idx: The index it should have.  WARNING: If this is already in the vocabulary, it will be overwritten.
        Returns:
            The index of the word after it was added (useful if you don't specify the index).
        """
        assert not self.final, "Vocabulary already finalized, cannot add more words"
        
        if idx == -1:
            idx = max(self._id_to_word.keys()) + 1

        self._id_to_word[idx] = word
        self._word_to_id[word] = idx

        return idx
    

    def lookup_word(self, idx: int) -> str:
        """
        Args:
            The integer index of a token
        Returns:
            It's string representation
        """
        
        return self._id_to_word.get(idx, None)

    def examples(self, limit:int = 20) -> Iterable[str]:
        """
        Get examples from the vocab.  To make them interesting, sort by length.

        Args:
            limit: How many to return
        Returns:
            A list of the elements
        """
        return sorted(self._word_to_id, key=len, reverse=True)[:limit]
    
    def lookup_index(self, word: str) -> int:
        """
        Args:
            The string representation of a word
        Returns:
            It's index in the lookup, returning the UNK token's lookup otherwise
        """
        
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            if self.final:
                return self._word_to_id[self._unk]
            else:
                return self.add(word)

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # Add code to generate the vocabulary that the vocab lookup
        # function can use!
        assert self._unk in self._word_to_id
        self.final = True

        logging.debug("%i vocab elements, including: %s" % (len(self), str(self.examples(10))))
            
class ToyTokenizerGuesser(Guesser):
    """Class that builds a vocabulary and then computes tf-idf scores
    given a corpus.

    """

    def __init__(self, max_vocab_size:int=10000,
                 whitespace_split:callable=TreebankWordTokenizer().tokenize):
        """
        Create a new tokenizer object

        Args:
           max_vocab_size: Do not let the vocab grow to be larger than this
           whitespace_split: Function that is used to split tokens by whitespace
        """
        
        self._max_vocab_size = max_vocab_size
        self._total_docs = 0

        self._whitespace_split = whitespace_split
        
        self._doc_vectors = None
        self._docs_final = False
        
        self._vocab = Vocab()
        self._end_id = self._vocab.add(kEND_STRING)
        self._inverse_vocab = {}

        self._docs = None
        self._labels = []

        # Add your code here if you need to!

    def frequent_bigram(self, token_ids: Iterable[int], min_frequency=2) -> Tuple[int, int]:
        """
        Return the most frequent byte pair, excluding pairs that
        contain the end of text token.

        So that the instructions are clear and there's no ambiguity
        (especially for the autograder), if there are equally frequent
        pairs, take the one with the lower *first* token.  If there are
        ties among pairs with the same first token, take the one with
        the lower *second* token.  This is easily accomplished with the
        'sorted' method.

        Args:
            token_ids: All of your tokens as represented by integers
            min_frequency: A pair of tokens must appear at least this many times to be returned
        Returns:
            The tuple of byte pairs that is most frequent in the input
        """

        # Complete this function!
        
        

        return None

    @staticmethod
    def merge_tokens(tokens: Iterable[int], merge_left: int, merge_right: int, merge_id: int) -> Iterable[int]:
        """Given a sequence of integers that represent byte pairs,

        replace all instances of a particular byte pair with a new
        symbol.

        Args:
            tokens: The original sequence
            merge_left: The left pair of the token pair to merge
            merge_right: The right pair of the token pair to merge
            merge_id: The new token to replace all of those instances with
        Returns:
            A sequence of integers where all of the (merge_left, merge_right) instances have been replaced by merge_id
        """
        replaced = tokens

        # Complete this function!
        


        return replaced    
    
    def train(self, training_data, answer_field='page', split_by_sentence=False) -> None:
        """
        Go through the training data, split them into tokens, build
        your tokenizer vocabulary, and then create your embeddings for
        the training data.

        Args:
            training_data: The raw data we will build dataset from
            answer_field: The field in the JSON that will become the guess when we find the closest document to a query
            split_by_sentence: Consider each of the sentences indepndently or not
        """
        # Extract the data into self.answers, self.questions
        Guesser.train(self, training_data, answer_field, split_by_sentence)

        #Complete this function!
        
        frequency = FreqDist()
        for question in (progress := tqdm(self.questions)):
            # This will create a whitespace vocab, but you should remove and replace
            # this code
            for word in self.whitespace_tokenize(question):
                 frequency[word] += 1
            progress.set_description("Creating initial vocabulary")
            


        # This code stub is here just so it will work before you
        # implement BPE training, remove it when you do.
        current_vocab_size = len(self._vocab)
        if current_vocab_size < 260:
            for word in frequency.most_common(self._max_vocab_size - current_vocab_size):
                self._vocab.add(word)
            
        self._vocab.finalize()
        assert len(self._vocab) < self._max_vocab_size
               
        for question in (progress := tqdm(self.questions)):
            progress.set_description("Creating document freq")
            self.scan_document(question)


        self.finalize_docs()
            
        assert self._total_docs == len(self.questions), "Number of documents mismatch"
        self._doc_vectors = np.zeros((len(self.questions), len(self._vocab)))
        for row_id, question in enumerate(progress := tqdm(self.questions)):
            progress.set_description("Creating document vecs")            
            self._doc_vectors[row_id] = self.embed(question)        

        
    def finalize_docs(self) -> None:
        """
        This is a separate function for ease of unit testing.  It is
        called when we're done adding documents to our dataset.
        """        
        self._docs_final = True
        
    def __call__(self, question: Dict, max_n_guesses:int=1) -> Dict[str, Union[str, float]]:
        """
        Given a question, find the closest document in the training set and
        return a dictionary with that guess.
        
        Before you start coding this, remember what this function did in the
        last homework: given a query, it needs to find the training item
        closest to the query.  To do that, you need to do three things: turn
        the query into a vector, compute the similarity of that vector with
        each row in the matrix, and return the metadata associated with that
        row.

        We've helped you out by structuring the code so that it should be easy
        for you to complete it.  \`question\_tfidf\` is the vector after you
        embed it.  This code is already done for you (assuming you've
        completed \`inv\_docfreq\` already).

        Then you'll need to go through the rows in \`self.\_doc\_vectors\` and
        find the closest row.  Call whatever the closest is \`best\` and
        return the appropriate metadata.  This is implemented for you already.

        Args:
            question: The query you need to find the answer toArgs:
            max_n_guesses: The number of guesses to return
        Returns:
            A list of the best guesses
        """
        
        assert max_n_guesses == 1, "We only support top guess"
        
        question_tfidf = self.embed(question).reshape(1, -1)

        # This code is wrong, you need to fix it!  You'll want to use "argmax" and perhapse "reshape"
        best = 0
        cosine = np.zeros(5)
        
        return [{"question": self.questions[best],
                 "guess": self.answers[best],
                 "confidence": cosine[best]}]
        

    def scan_document(self, text: str) -> int:
        """
        Tokenize a piece of text and compute the document frequencies.

        Args:
            text: The raw string containing a document

        Returns:
            The new number of documents 
        """

        assert self._vocab.final, "scan_document can only be run with finalized vocab"
        assert not self._docs_final, "scan_document can only be run with non-finalized doc counts"

        # Complete this function!
        
        tokenized = list(self.tokenize(text))
        if len(tokenized) == 0:
            logging.warning("Empty doc: %30s, tokenize: %30s, vocab: %30s" % (text, str(tokenized), " ".join(self._vocab.examples(5))))

        self._total_docs += 1

        return self._total_docs
        
    def embed(self, text: str) -> npt.NDArray[np.float64]:
        """
        Given a document, create a vector representation of its tfidf
        vector, where each vector corresponds to a token in your
        vocabulary.

        Args:
            text: A string of text

        Returns:
            A vector where each dimensions corresponds to the weight of the vector
        """
        
        # You don't need to modify this code
        vector = np.zeros(len(self._vocab))
        doc_frequency = FreqDist(x for x in self.tokenize(text) if x in self._vocab)

        for word in doc_frequency:
            vector[word] = doc_frequency.freq(word) * self.inv_docfreq(word)
        return vector

    def whitespace_tokenize(self, sent: str) -> Iterable[int]:
        """
        A function that takes the initial string and turns it into
        integers where each integer is a whitespace delimited token.

        By default, the tokenization is based on the Treebank tokenizer.

        Args:
            sent: A string of text
        Returns:
            A list of the integer ids
        """
        for chunk in self._whitespace_split(sent):
            yield self._vocab.lookup_index(chunk)

    def initial_tokenize(self, sent: str) -> Iterable[int]:
        """
        Function to map individual bytes to integers, adding a
        "END OF STRING" token to the end.  This is the foundation that
        you can then later do BPE on.

        Args:
            sent: A string of text
        Returns:
            A list of the integer ids
        """

        # You do not need to modify this code
        chars = list(map(int, bytearray(sent, "utf-8")))
        chars += [self._end_id]
        return chars
    
    def tokenize(self, sent: str) -> Iterable[int]:
        """
        Return a generator over tokens in the sentence and add the
        "END OF STRING" token, <|ENDOFTEXT|> token id to the end of the
        string.

        Args:
            sent: A string of text
        Returns:
            A list of the integer ids
        """
        assert self._vocab.final
        token_ids = self.initial_tokenize(sent)

        # Finish this function!
        

        

            

            
        
        return self.whitespace_tokenize(sent)
        

    def inv_docfreq(self, word: int) -> float:
        """Compute the inverse document frequency of a word in log base

        10.  Return 1.0 if we didn't see the word index in training
        (however, this should never happen in normal operation).

        Because we may have terms that have never been seen, add 1 to
        all of the document counts (in other words, we assume that a
        term appears at least once).

        Keyword arguments:
        word -- The word to look up the document frequency of a word.
        """
        assert self._docs_final, "Documents must be finalized"

        return 1.0
        

if __name__ == "__main__":
    # Load a tf-idf guesser and run it on some questions
    from params import *
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    flags.guesser_type = "ToyTokenizer"
    
    guesser = load_guesser(flags)
    guesser.train(kTOY_DATA["train"], answer_field='page', split_by_sentence=False)

    logging.debug("Document matrix is %i by %i, has %i non-zero entries" %
                      (guesser._doc_vectors.shape[0],
                       guesser._doc_vectors.shape[1],
                       np.count_nonzero(guesser._doc_vectors)))

    for query_row in kTOY_DATA["dev"]:
        query = query_row["text"]
        print("----------------------")
        guess = guesser(query)
        print(query, guess)
