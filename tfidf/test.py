import unittest
from math import log

from vocab import Vocabulary, kUNK, log10


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.unk_cutoff = 2        
        self.vocab = Vocabulary(unk_cutoff=self.unk_cutoff)

    def test_vocab(self):
        self.vocab.train_seen("a", 300)

        self.vocab.train_seen("b")
        self.vocab.train_seen("c")
        self.vocab.finalize()

        # Infrequent words should look the same
        self.assertEqual(self.vocab.vocab_lookup("b"),
                         self.vocab.vocab_lookup("c"))

        # Infrequent words should look the same as never seen words
        self.assertEqual(self.vocab.vocab_lookup("b"),
                         self.vocab.vocab_lookup("d"),
                         "")

        # The frequent word should be different from the infrequent word
        self.assertNotEqual(self.vocab.vocab_lookup("a"),
                            self.vocab.vocab_lookup("b"))

    def test_censor(self):
        self.vocab.train_seen("a", 300)

        self.vocab.train_seen("b")
        self.vocab.train_seen("c")
        self.vocab.finalize()

        censored_a = [str(x) for x in self.vocab.tokenize("a b d")]
        censored_b = [str(x) for x in self.vocab.tokenize("d b a")]
        censored_c = [str(x) for x in self.vocab.tokenize("a b d")]
        censored_d = [str(x) for x in self.vocab.tokenize("b d a")]

        self.assertEqual(censored_a, censored_c)
        self.assertEqual(censored_b, censored_d)

        # Should add start and end tag
        print(censored_a)
        self.assertEqual(len(censored_a), 3)
        self.assertEqual(censored_a[0], censored_b[2])
        self.assertEqual(censored_a[1], censored_b[0])

    def test_tf(self):
        self.vocab.train_seen("a", 300)
        self.vocab.finalize()

        self.vocab.add_document("a a b")

        # Test MLE
        word_a = self.vocab.vocab_lookup("a")
        word_b = self.vocab.vocab_lookup("b")
        word_c = self.vocab.vocab_lookup("c")

        self.assertAlmostEqual(self.vocab.term_freq(word_a), 0.66666666)
        self.assertAlmostEqual(self.vocab.term_freq(word_b), 0.33333333)
        self.assertAlmostEqual(self.vocab.term_freq(word_c), 0.33333333)

    def test_df(self):
        self.vocab.train_seen("a", 300)
        self.vocab.train_seen("b", 100)
        self.vocab.finalize()

        self.vocab.add_document("a a b")
        self.vocab.add_document("b b c")        
        self.vocab.add_document("a a a")
        self.vocab.add_document("a a a")                
        
        # Test MLE
        word_a = self.vocab.vocab_lookup("a")
        word_b = self.vocab.vocab_lookup("b")
        word_c = self.vocab.vocab_lookup("c")
        word_d = self.vocab.vocab_lookup("d")

        self.assertAlmostEqual(self.vocab.inv_docfreq(word_a), log10(1.3333333))
        self.assertAlmostEqual(self.vocab.inv_docfreq(word_b), log10(2.0))
        self.assertAlmostEqual(self.vocab.inv_docfreq(word_c), log10(4.0))
        self.assertAlmostEqual(self.vocab.inv_docfreq(word_d), log10(4.0))
        

if __name__ == '__main__':
    unittest.main()
