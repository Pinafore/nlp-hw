import unittest
from math import log

from guesser import kTOY_DATA
from toytokenizer_guesser import ToyTokenizerGuesser, kUNK, log10
from toytokenizer_guesser import kEND_STRING, Vocab

utf8_answers = {"ascii": [64, 35, 36, 37, 94, 38, 42, 42, 42, 40, 41, 257],
                "unicode": [208, 163, 208, 147, 206, 165, 206, 147, 206, 163, 257],
                "chinese": [233, 169, 172, 233, 135, 140, 229, 133, 176, 230, 156, 137, 229, 190, 136, 229, 176, 145, 231, 154, 132, 233, 169, 172, 229, 140, 185, 44, 229, 174, 189, 229, 186, 166, 228, 187, 133, 230, 156, 137, 50, 48, 48, 229, 133, 172, 233, 135, 140, 44, 229, 143, 175, 230, 152, 175, 233, 169, 172, 233, 135, 140, 229, 133, 176, 230, 156, 137, 229, 190, 136, 229, 164, 154, 229, 133, 176, 257],
                "elements": [65, 110, 100, 32, 97, 114, 103, 111, 110, 44, 32, 107, 114, 121, 112, 116, 111, 110, 44, 32, 110, 101, 111, 110, 44, 32, 114, 97, 100, 111, 110, 44, 32, 120, 101, 110, 111, 110, 44, 32, 122, 105, 110, 99, 44, 32, 97, 110, 100, 32, 114, 104, 111, 100, 105, 117, 109, 44, 257]}

frequent = {"ascii": (42, 42),
            "chinese": (229, 133),
            "unicode": (147, 206),
            "elements": (44, 32)}

element_ref = {0: 'And argon!krypton!neon!radon!xenon!zinc!and rhodium,<ENDOFTEXT>',
               1: 'And argo@krypto@neo@rado@xeno@zinc!and rhodium,<ENDOFTEXT>',
               2: 'And arg#krypt#ne#rad#xen#zinc!and rhodium,<ENDOFTEXT>',
               3: 'An$arg#krypt#ne#rad#xen#zinc!an$rhodium,<ENDOFTEXT>',
               4: 'A%arg#krypt#ne#rad#xen#zinc!a%rhodium,<ENDOFTEXT>'}
    
class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):  
        self.guesser = ToyTokenizerGuesser()
        self.test_strings = {"elements": "And argon, krypton, neon, radon, xenon, zinc, and rhodium,",
                            "unicode": "УГΥΓΣ",
                            "chinese": "马里兰有很少的马匹,宽度仅有200公里,可是马里兰有很多兰",
                            "ascii": "@#$%^&***()"}

    def test_initial(self):
        for test in self.test_strings:
            self.assertEqual(list(self.guesser.initial_tokenize(self.test_strings[test])),
                             utf8_answers[test], test)

    def test_frequent(self):
        for test in self.test_strings:
            self.assertEqual(frequent[test], self.guesser.frequent_bigram(utf8_answers[test]), test)

    def test_chinese_vocab_extra_credit(self):
        """
        This is for extra credit, you're not required to implement this.
        """
        training_data = [{"text": self.test_strings['chinese'], "page": ""}]
        self.guesser.train(training_data)

        print(self.guesser._vocab._word_to_id.keys())
            
    def test_train_vocab(self):
        self.guesser.train(kTOY_DATA["train"], answer_field='page', split_by_sentence=False)

        self.assertEqual(self.guesser._vocab.examples(3),
                         ['which states that bad money drives ',
                          'For 10 points, name this author of ',
                          'New England state with capital at '])

    def test_no_end_in_frequent(self):
        import random
        
        end = self.guesser._end_id
        sample = [end] * 20 + [5] * 20
        random.shuffle(sample)
        sample += [5] * 2
            
        self.assertEqual(self.guesser.frequent_bigram(sample), (5, 5))

        sample = [end, 5] * 20
        self.assertEqual(self.guesser.frequent_bigram(sample), None)

    @staticmethod
    def rerender(vocab, tokens, separator=""):
      result = []
      for token in tokens:
        token_string = vocab.lookup_word(token)
        assert token_string is not None, "Vocab lookup failed for %i" % token
        result.append(token_string)
      return separator.join(result)
        
    def test_replace(self):
        import string
        
        vocab = Vocab()
        vocab.add(kEND_STRING)

        token_sequence = utf8_answers["elements"]

        merges = [", ", "n!", "o@", "d ", "n$"]

        print(token_sequence)
        print(self.rerender(vocab, token_sequence))            
        for ii, replacement in enumerate("!@#$%"):
            top = self.guesser.frequent_bigram(token_sequence)
            self.assertNotEqual(top, None)
            freq_left, freq_right = top
            idx = vocab.add_from_merge(freq_left, freq_right)
            token_sequence = self.guesser.merge_tokens(token_sequence, freq_left, freq_right, idx)
            merge_found = vocab.lookup_word(freq_left) + vocab.lookup_word(freq_right)
            self.assertEqual(merge_found, merges[ii],
                             "Disagree on merge: %s vs %s" % (merge_found, merges[ii]))
            vocab.add(replacement, idx)

            pretty = self.rerender(vocab, token_sequence)            
            print(token_sequence)            
            print(pretty)

            self.assertEqual(element_ref[ii], pretty)

               

    def test_embed(self):
        self.guesser.train(kTOY_DATA["tiny"], answer_field='page', split_by_sentence=False)

        test_doc = self.guesser.embed("currency Brazil")
        currency = self.guesser._vocab.lookup_index("currency")
        brazil = self.guesser._vocab.lookup_index("Brazil")
        
        # Test that b and c have same frequency (Unknown tokens)
        self.assertAlmostEqual(test_doc[currency], 0.0)
        self.assertAlmostEqual(test_doc[brazil], 0.0)

    def test_empty_df(self):
        self.guesser._vocab.finalize()

        self.guesser.scan_document("aab")
        self.guesser.scan_document("bbc")        
        self.guesser.scan_document("aaa")
        self.guesser.scan_document("aaa")

        self.guesser.finalize_docs()
        
        # Test inverse doc frequency
        word_a = self.guesser._vocab.lookup_index("a")
        word_b = self.guesser._vocab.lookup_index("b")
        word_c = self.guesser._vocab.lookup_index("c")
        word_d = self.guesser._vocab.lookup_index("d")

        self.assertAlmostEqual(self.guesser.inv_docfreq(word_a), 0.12494, delta=0.01)
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_b), 0.30103, delta=0.01)
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_c), 0.60206, delta=0.01)
        self.assertAlmostEqual(self.guesser.inv_docfreq(word_d), 0.0, delta=0.01)

    def test_tokenize_wo_merge(self):
        """
        If we don't train the tokenizer, the tokenization should just be characters.
        """
        self.guesser._vocab.finalize()
        reference = ["T*h*i*s* *c*a*p*i*t*a*l* *o*f* *E*n*g*l*a*n*d*.*<ENDOFTEXT>",
                     "T*h*e* *a*u*t*h*o*r* *o*f* *P*r*i*d*e* *a*n*d* *P*r*e*j*u*d*i*c*e*.*<ENDOFTEXT>",
                     "T*h*e* *c*o*m*p*o*s*e*r* *o*f* *t*h*e* *M*a*g*i*c* *F*l*u*t*e*.*<ENDOFTEXT>",
                     "T*h*e* *e*c*o*n*o*m*i*c* *l*a*w* *t*h*a*t* *s*a*y*s* *'*g*o*o*d* *m*o*n*e*y* *d*r*i*v*e*s* *o*u*t* *b*a*d*'*.*<ENDOFTEXT>",
                     "L*o*c*a*t*e*d* *o*u*t*s*i*d*e* *B*o*s*t*o*n*,* *t*h*e* *o*l*d*e*s*t* *U*n*i*v*e*r*s*i*t*y* *i*n* *t*h*e* *U*n*i*t*e*d* *S*t*a*t*e*s*.*<ENDOFTEXT>"]
        
        for doc_id, doc in enumerate(kTOY_DATA["dev"]):
            tokens = self.guesser.tokenize(doc["text"])
            print("@@@", doc["text"], list(tokens))
            reconstruction = self.rerender(self.guesser._vocab, tokens, "*")
            self.assertEqual(reconstruction, reference[doc_id], "Tokenization of %s bad results %s!=%s." % (doc["text"], reconstruction, reference[doc_id]))
            
    def test_tokenize(self):
        """
        Test tokenization after training the tokenizer.
        """
        self.guesser.train(kTOY_DATA["train"], answer_field='page', split_by_sentence=False)

        reference = ['This *capital *of *En*gl*an*d*.*<ENDOFTEXT>',
                     'Th*e *au*thor of *Pr*ide* *and *Prejudice.*<ENDOFTEXT>',
                     'Th*e *composer *of *the *Ma*gic Flut*e.*<ENDOFTEXT>',
                     "Th*e *e*co*n*om*ic la*w *that *s*a*y*s *'*goo*d money *dr*ives *out *b*a*d*'*.*<ENDOFTEXT>",
                     "L*o*c*ate*d *ou*t*s*ide* *B*o*st*on*, *the *oldest *Un*iversit*y in *the United State*s.*<ENDOFTEXT>"]
        
        for doc_id, doc in enumerate(kTOY_DATA["dev"]):
            tokens = self.guesser.tokenize(doc["text"])
            print("!!!", doc["text"], list(tokens))
            reconstruction = self.rerender(self.guesser._vocab, tokens, "*")
            self.assertNotEqual(tokens, [])
            self.assertEqual(reconstruction, reference[doc_id], "Tokens: %s" % str(tokens))
        
    def test_df(self):
        self.guesser.train(kTOY_DATA["tiny"], answer_field='page', split_by_sentence=False)

        england = self.guesser._vocab.lookup_index("England")
        russia = self.guesser._vocab.lookup_index("Russia")
        oov = self.guesser._vocab.lookup_index("z")        

        self.assertAlmostEqual(self.guesser.inv_docfreq(england), 0.0)
        self.assertAlmostEqual(self.guesser.inv_docfreq(russia), 0.0)
        self.assertAlmostEqual(self.guesser.inv_docfreq(oov), 0.0)
        
if __name__ == '__main__':
    unittest.main()
