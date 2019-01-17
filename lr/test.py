import unittest
import json

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

from lr import QuizBowlData, setup, accuracy

kTOY_TRAIN = json.loads("""[{"text": "treatise author bacon", "category": "Philosophy"}, 
                            {"text": "treatise author kant", "category": "Philosophy"},
                            {"text": "treatise author leibnitz", "category": "Philosophy"},
                            {"text": "novel author dickens", "category": "Literature"}, 
                            {"text": "novel author swift", "category": "Literature"},
                            {"text": "novel author twain", "category": "Literature"}]""")

kTOY_TEST = json.loads("""[{"text": "treatise author spinoza", "category": "Philosophy"}, 
                           {"text": "novel poem author tagore", "category": "Literature"}]""" )

class TestTorchLogReg(unittest.TestCase):
    def setUp(self):
        self.vectorizer = TfidfVectorizer(max_df=0.5, max_features=10,
                                          min_df=2, stop_words='english',
                                          use_idf=True)

        self.train = QuizBowlData(None, "", self.vectorizer)
        self.train.vectorize(kTOY_TRAIN)

        self.test = QuizBowlData(None, "", self.vectorizer, is_train=False)
        self.test.vectorize(kTOY_TEST)
        
        self.model = nn.Linear(2, 2)
        self.model.bias.data.fill_(0.0)
        self.model.weight.data.fill_(0.0)
        self.model.bias.data[0] = -1
        self.model.weight.data[0][0] = 1
        self.model.weight.data[0][1] = -2
        self.model.weight.data[1][0] = -3
        self.model.weight.data[1][1] = 2
        
    def testVectorize(self):
        """
        Test that the data is represented correctly.
        """
        
        doc = self.train.tfidf

        self.assertEqual(doc[0][self.vectorizer.vocabulary_["treatise"]], 1)
        self.assertEqual(doc[0][self.vectorizer.vocabulary_["novel"]], 0)
        self.assertEqual(doc[1][self.vectorizer.vocabulary_["treatise"]], 1)
        self.assertEqual(doc[1][self.vectorizer.vocabulary_["novel"]], 0)
        self.assertEqual(doc[2][self.vectorizer.vocabulary_["treatise"]], 1)
        self.assertEqual(doc[2][self.vectorizer.vocabulary_["novel"]], 0)

        self.assertEqual(doc[3][self.vectorizer.vocabulary_["treatise"]], 0)
        self.assertEqual(doc[3][self.vectorizer.vocabulary_["novel"]], 1)
        self.assertEqual(doc[4][self.vectorizer.vocabulary_["treatise"]], 0)
        self.assertEqual(doc[4][self.vectorizer.vocabulary_["novel"]], 1)
        self.assertEqual(doc[5][self.vectorizer.vocabulary_["treatise"]], 0)
        self.assertEqual(doc[5][self.vectorizer.vocabulary_["novel"]], 1)

        doc = self.test.tfidf
        self.assertEqual(doc[0][self.vectorizer.vocabulary_["treatise"]], 1)
        self.assertEqual(doc[0][self.vectorizer.vocabulary_["novel"]], 0)

        self.assertEqual(doc[1][self.vectorizer.vocabulary_["treatise"]], 0)
        self.assertEqual(doc[1][self.vectorizer.vocabulary_["novel"]], 1)

    def testSetup(self):
        """
        Test that the dimension of the parameters are correct.
        """
        
        train = QuizBowlData(None, "", self.vectorizer)
        train.vectorize(kTOY_TRAIN)        
        model, optimizer = setup(train, 1.0)

        self.assertEqual(list(model.weight.size()), [2, 2])
        self.assertEqual(list(model.bias.size()), [2])

    def testAccuracy(self):
        """
        Return a float that had the proportion of examples with correct predictions.
        """
        
        loader = torch.utils.data.DataLoader(dataset=self.test, 
                                             shuffle=False)
        acc = accuracy(self.model, loader)
        self.assertEqual(acc, 1.0)
        print(acc)
        
if __name__ == '__main__':
    unittest.main()
