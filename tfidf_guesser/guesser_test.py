
import unittest
from guesser import kTOY_DATA
from tfidf_guesser import TfidfGuesser, kTFIDF_TEST_QUESTIONS

class GuesserTest(unittest.TestCase):
    def setUp(self):
        self.guesser = TfidfGuesser("data/test_guesser", min_df=0.0, max_df=1.0)
        
        self.guesser.train(kTOY_DATA["train"], 'page', False, 0, -1)

        print("VOCAB")
        for v, k in sorted( ((v,k) for k,v in self.guesser.tfidf_vectorizer.vocabulary_.items())):
            print("%5i %20s" % (v, k))

    def test_stopwords(self):
        self.assertTrue("the" not in self.guesser.tfidf_vectorizer.vocabulary_)

    def test_length(self):
        self.assertEqual(len(self.guesser.answers), 13)
        
        new_guesser = TfidfGuesser("data/test_guesser", min_df=0.0, max_df=1.0)
        new_guesser.train(kTOY_DATA["train"], min_length=60, max_length=90)

        self.assertEqual(len(new_guesser.answers), 7)
        

    def test_top_single(self):
        print("INDIVIDUAL")
        for query_result in kTOY_DATA["dev"]:
            query = query_result["text"]
            top = query_result["top"]
            second = query_result["second"]
            
            guesses = self.guesser(query)

            print("%60s %30s %30s %0.3f" % (query[:60], top[:30], guesses[0]['guess'][:30], guesses[0]['confidence']))
            print("%60s %30s %30s %0.3f" % ("", "", guesses[1]['guess'][:30], guesses[1]['confidence']))
            self.assertEqual(guesses[0]['guess'], top)
            self.assertEqual(guesses[1]['guess'], second)

    def test_top_batch(self):
        print("BATCH")
        questions = list(kTFIDF_TEST_QUESTIONS)
        guesses = self.guesser.batch_guess(questions, 2, block_size=3)
        for query_result, guess in zip(kTOY_DATA["dev"], guesses):
            query = query_result["text"]
            top = query_result["top"]
            second = query_result["second"]

            self.assertEqual(guess[0]['guess'], top)
            self.assertEqual(guess[1]['guess'], second)

            
if __name__ == '__main__':
    unittest.main()
