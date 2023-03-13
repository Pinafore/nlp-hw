
import unittest
from tfidf_guesser import TfidfGuesser

class GuesserTest(unittest.TestCase):
    def setUp(self):
        self.guesser = TfidfGuesser("data/test_guesser", min_df=0.0, max_df=1.0)
        self.toy_data = [{'page': 'Maine', 'text': 'For 10 points, name this New England state with capital at Augusta.'},
                         {'page': 'Massachusetts', 'text': 'For ten points, identify this New England state with capital at Boston.'},
                         {'page': 'Boston', 'text': 'For 10 points, name this city in New England, the capital of Massachusetts.'},
                         {'page': 'Jane_Austen', 'text': 'For 10 points, name this author of Pride and Prejudice.'},
                         {'page': 'Jane_Austen', 'text': 'For 10 points, name this author of Emma and Pride and Prejudice.'},
                         {'page': 'Wolfgang_Amadeus_Mozart', 'text': 'For 10 points, name this composer of Magic Flute and Don Giovanni.'},
                         {'page': 'Wolfgang_Amadeus_Mozart', 'text': 'Name this composer who wrote a famous requiem and The Magic Flute.'},
                         {'page': "Gresham's_law", 'text': 'For 10 points, name this economic principle which states that bad money drives good money out of circulation.'},
                         {'page': "Gresham's_law", 'text': "This is an example -- for 10 points \\-- of what Scotsman's economic law, which states that bad money drives out good?"},
                         {'page': "Gresham's_law", 'text': 'FTP name this economic law which, in simplest terms, states that bad money drives out the good.'},
                         {'page': 'Rhode_Island', 'text': "This colony's Touro Synagogue is the oldest in the United States."},
                         {'page': 'Lima', 'text': 'It is the site of the National University of San Marcos, the oldest university in South America.'},
                         {'page': 'College_of_William_&_Mary', 'text': 'For 10 points, identify this oldest public university in the United States, a college in Virginia named for two monarchs.'}]

        self.queries = {"This capital of England": ['Maine', 'Boston'],
                        "The author of Pride and Prejudice": ['Jane_Austen', 'Jane_Austen'],
                        "The composer of the Magic Flute": ['Wolfgang_Amadeus_Mozart', 'Wolfgang_Amadeus_Mozart'],
                        "The economic law that says 'good money drives out bad'": ["Gresham's_law", "Gresham's_law"],
                        "located outside Boston, the oldest University in the United States": ['College_of_William_&_Mary', 'Rhode_Island']}
        
        self.guesser.train(self.toy_data, 'page', False, 0, -1)

    def test_length(self):
        self.assertEqual(len(self.guesser.answers), 13)
        
        new_guesser = TfidfGuesser("data/test_guesser", min_df=0.0, max_df=1.0)
        new_guesser.train(self.toy_data, min_length=60, max_length=90)

        self.assertEqual(len(new_guesser.answers), 7)
        

    def test_top_single(self):
        print("INDIVIDUAL")
        for query in self.queries:
            top, second = self.queries[query]
            guesses = self.guesser(query)
            print("%60s %30s %30s %0.3f" % (query[:60], top[:30], guesses[0]['guess'][:30], guesses[0]['confidence']))
            self.assertEqual(guesses[0]['guess'], top)
            self.assertEqual(guesses[1]['guess'], second)

    def test_top_batch(self):
        print("BATCH")
        questions = list(self.queries.keys())
        guesses = self.guesser.batch_guess(questions, 2, block_size=3)
        for query, guesses in zip(questions, guesses):
            top, second = self.queries[query]
            print("%60s %30s %30s %0.3f" % (query[:60], top[:30], guesses[0]['guess'][:30], guesses[0]['confidence']))
            self.assertEqual(guesses[0]['guess'], top)
            self.assertEqual(guesses[1]['guess'], second)

            
if __name__ == '__main__':
    unittest.main()
