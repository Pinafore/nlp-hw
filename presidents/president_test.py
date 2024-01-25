import unittest
from president_guesser import *

class TestPresidentGuessers(unittest.TestCase):
    def setUp(self):
        self.reference = kPRESIDENT_DATA['dev']
        self.pg = PresidentGuesser()
        self.pg.train(kPRESIDENT_DATA['train'])

    def test_basic(self):
        for ii in self.reference:
            guess = self.pg(ii["text"])[0]["guess"]
            self.assertEqual(guess, ii["page"], "Wrong answer for: %s" % ii["text"])

if __name__ == '__main__':
    unittest.main()
