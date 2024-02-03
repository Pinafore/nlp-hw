import unittest
from math import log

from toytfidf_guesser import ToyTfIdfGuesser, kUNK, log10


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.unk_cutoff = 2        
        self.guesser = ToyTfIdfGuesser(filename="models/TfidfGuesser", unk_cutoff=self.unk_cutoff)

    def test_vocab(self):
        self.guesser.vocab_seen("a", 300)

        self.guesser.vocab_seen("b")
        self.guesser.vocab_seen("c")
        self.guesser.finalize_vocab()

        # Infrequent words should look the same
        self.assertEqual(self.guesser.vocab_lookup("b"),
                         self.guesser.vocab_lookup("c"))

        # Infrequent words should look the same as never seen words
        self.assertEqual(self.guesser.vocab_lookup("b"),
                         self.guesser.vocab_lookup("d"),
                         "")

        # The frequent word should be different from the infrequent word
        self.assertNotEqual(self.guesser.vocab_lookup("a"),
                            self.guesser.vocab_lookup("b"))

    def test_censor(self):
        self.guesser.vocab_seen("a", 300)

        self.guesser.vocab_seen("b")
        self.guesser.vocab_seen("c")
        self.guesser.finalize_vocab()

        censored_a = [str(x) for x in self.guesser.tokenize("a b d")]
        censored_b = [str(x) for x in self.guesser.tokenize("d b a")]
        censored_c = [str(x) for x in self.guesser.tokenize("a b d")]
        censored_d = [str(x) for x in self.guesser.tokenize("b d a")]

        self.assertEqual(censored_a, censored_c)
        self.assertEqual(censored_b, censored_d)

        # Should add start and end tag
        print(censored_a)
        self.assertEqual(len(censored_a), 3)
        self.assertEqual(censored_a[0], censored_b[2])
        self.assertEqual(censored_a[1], censored_b[0])

    def test_tf(self):
        self.guesser.vocab_seen("a", 300)
        self.guesser.finalize_vocab()

        self.guesser.scan_document("a a b")

        # Test MLE
        word_a = self.guesser.vocab_lookup("a")
        word_b = self.guesser.vocab_lookup("b")
        word_c = self.guesser.vocab_lookup("c")

        # Test that b and c have same frequency (Unknown tokens)
        self.assertAlmostEqual(self.guesser.global_freq(word_a), 0.66666666)
        self.assertAlmostEqual(self.guesser.global_freq(word_b), 0.33333333)
        self.assertAlmostEqual(self.guesser.global_freq(word_c), 0.33333333)

    def test_df(self):
        self.guesser.vocab_seen("a", 300)
        self.guesser.vocab_seen("b", 100)
        self.guesser.finalize_vocab()

        self.guesser.scan_document("a a b")
        self.guesser.scan_document("b b c")        
        self.guesser.scan_document("a a a")
        self.guesser.scan_document("a a a")

        self.guesser.finalize_docs()
        
        # Test inverse doc frequency
        word_a = self.guesser.vocab_lookup("a")
        word_b = self.guesser.vocab_lookup("b")
        word_c = self.guesser.vocab_lookup("c")
        word_d = self.guesser.vocab_lookup("d")

        self.assertAlmostEqual(self.guesser.inv_docfreq(word_a), log10(1.3333333))
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_b), log10(2.0))
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_c), log10(4.0))
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_d), log10(4.0))
        

if __name__ == '__main__':
    unittest.main()
