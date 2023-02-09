import argparse
import json
from collections import defaultdict
from math import log
import os
from progress.bar import Bar

from typing import Iterable, Tuple, Dict

from nltk.tokenize import TreebankWordTokenizer
from nltk import FreqDist

kUNK = "<UNK>"

def log10(x):
    return log(x) / log(10)

def lower(str):
    return str.lower()


class TfIdf:
    """Class that builds a vocabulary and then computes tf-idf scores
    given a corpus.

    """

    def __init__(self, vocab_size=10000,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lower, unk_cutoff=2):
        self._vocab_size = vocab_size
        self._total_docs = 0

        self._vocab_final = False
        self._vocab = {}
        self._unk_cutoff = unk_cutoff

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function

        # Add your code here!

    def train_seen(self, word: str, count: int=1):
        """Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.

        word -- The string represenation of the word.  After we
        finalize the vocabulary, we'll be able to create more
        efficient integer representations, but we can't do that just
        yet.

        count -- How many times we've seen this word (by default, this is one).
        """

        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!

    def add_document(self, text: str):
        """
        Tokenize a piece of text and add the entries to the class's counts.

        text -- The raw string containing a document
        """

        assert self._vocab_final, "add_document can only be run with finalized vocab"

        for word in self.tokenize(text):
            None

    def tokenize(self, sent: str) -> Iterable[int]:
        """Return a generator over tokens in the sentence; return the vocab
        of a sentence if finalized, otherwise just return the raw string.

        sent -- A string

        """

        # You don't need to modify this code.
        for ii in self._tokenizer(sent):
            if self._vocab_final:
                yield self.vocab_lookup(ii)
            else:
                yield ii

    def doc_tfidf(self, doc: str) -> Dict[Tuple[str, int], float]:
        """Given a document, create a dictionary representation of its tfidf vector

        doc -- raw string of the document"""

        counts = FreqDist(self.tokenize(doc))
        d = {}
        for ii in self._tokenizer(doc):
            ww = self.vocab_lookup(ii)
            d[(ww, ii)] = counts.freq(ww) * self.inv_docfreq(ww)
        return d
                
    def term_freq(self, word: int) -> float:
        """Return the frequence of a word if it's in the vocabulary, zero otherwise.

        word -- The integer lookup of the word.
        """

        return 0.0

    def inv_docfreq(self, word: int) -> float:
        """Compute the inverse document frequency of a word.  Return 0.0 if
        the word has never been seen.

        Keyword arguments:
        word -- The word to look up the document frequency of a word.

        """

        return 0.0

    def vocab_lookup(self, word: str) -> int:
        """
        Given a word, provides a vocabulary integer representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.

        This is useful for turning words into features in later homeworks.

        word -- The word to lookup
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._vocab[kUNK]

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # Add code to generate the vocabulary that the vocab lookup
        # function can use!

        self._vocab_final = True



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--root_dir", help="QB Dataset for training",
                           type=str, default='../data',
                           required=False)
    argparser.add_argument("--train_dataset", help="QB Dataset for training",
                           type=str, default='qanta.train.json',
                           required=False)
    argparser.add_argument("--example", help="What answer we use for testing",
                           type=str, default='Australia',
                           required=False)    
    argparser.add_argument("--limit", help="Number of training documents",
                           type=int, default=-1, required=False)
    args = argparser.parse_args()

    vocab = TfIdf()

    with open(os.path.join(args.root_dir, args.train_dataset)) as infile:
        data = json.load(infile)["questions"]
        if args.limit > 0:
            data = data[:args.limit]
        bar = Bar('Create vocab', max=len(data))
        for ii in data:
            bar.next()
            for word in vocab.tokenize(ii["text"]):
                vocab.train_seen(word)
        vocab.finalize()
        bar.finish()

        bar = Bar('Create counts', max=len(data))        
        for ii in data:
            vocab.add_document(ii["text"])
            bar.next()
        bar.finish()

    with open(os.path.join(args.root_dir, args.train_dataset)) as infile:
        data = json.load(infile)["questions"]
        bar = Bar('Heldout score', max=len(data))
        example = ""
        for ii in data:
            bar.next()
            if ii["page"] == args.example:
                example += " %s " % ii["text"]
        bar.finish()

        vector = vocab.doc_tfidf(example)
        for word, tfidf in sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            print("%s:%i\t%f" % (word[1], word[0], tfidf))
        for word, tfidf in sorted(vector.items(), key=lambda kv: kv[1], reverse=False)[:50]:
            print("%s:%i\t%f" % (word[1], word[0], tfidf))
