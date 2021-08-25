import unittest
from math import log

class StubDatabase:
    def __init__(self):
        self.guess_train_questions = []

    def add(self, question):
        self.guess_train_questions.append(question)

class StubQuestion:
    def __init__(self, question, answer):
        self.text = question
        self.page = answer
        
class TestGuesser(unittest.TestCase):

    def setUp(self):
        from tfidf_guesser import TfidfGuesser        
        self.guesser = TfidfGuesser()
        self.data = StubDatabase()
        self.num_duplicates = 5
        for ii in range(self.num_duplicates):
            self.data.add(StubQuestion("Who is buried in Grant's tomb?", "Ulysses_S._Grant"))
            self.data.add(StubQuestion("Who is buried in the Mausoleum at Halicarnassus?", "Mausolus"))

        self.guesser.train(self.data)

    def testSame(self):
        guess = self.guesser.guess(["Who is buried in Grant's tomb?"], max_n_guesses=1)
        self.assertEqual(guess[0][0][0], "Ulysses_S._Grant")

        guess = self.guesser.guess(["Who is buried in the Mausoleum at Halicarnassus?"], max_n_guesses=1)
        self.assertEqual(guess[0][0][0],  "Mausolus")

    def testSimilar(self):
        guess = self.guesser.guess(["tomb"], max_n_guesses=self.num_duplicates + 1)
        for ii in range(self.num_duplicates):
            self.assertEqual(guess[0][ii][0], "Ulysses_S._Grant")
            self.assertEqual(guess[0][ii][0], guess[0][0][0])
            
        self.assertEqual(guess[0][self.num_duplicates][0], "Mausolus")

        self.assertGreater(guess[0][0][1], guess[0][self.num_duplicates][1])

if __name__ == '__main__':
    unittest.main()
