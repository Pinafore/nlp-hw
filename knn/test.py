import unittest

from numpy import array

from knn import *

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.x = array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4],
                        [0, 2], [3, 2], [4, 2], [5, 2], [7, 3], [5, 5]])
        self.y = array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1])
        self.knn = {}
        for ii in [1, 2, 3]:
            self.knn[ii] = Knearest(self.x, self.y, ii, metric='cosine')

        self.queries = [array(x).reshape(1, -1) for x in
                        [[1, 5], [0, 3], [6, 1], [6, 4]]]
        self.test_y = [1, -1, 1, -1]

    def test1(self):
        self.assertEqual(self.knn[1].classify(self.queries[0]), 1)
        self.assertEqual(self.knn[1].classify(self.queries[1]), -1)
        self.assertEqual(self.knn[1].classify(self.queries[2]), 1)
        self.assertEqual(self.knn[1].classify(self.queries[3]), -1)
        self.assertEqual(self.knn[1].acccuracy(self.knn[1].confusion_matrix(self.queries, self.test_y)), 1.0)

    def test2(self):
        self.assertEqual(self.knn[2].classify(self.queries[0]), 1)
        self.assertEqual(self.knn[2].classify(self.queries[1]), -1)
        self.assertEqual(self.knn[2].classify(self.queries[2]), 1)
        self.assertEqual(self.knn[2].classify(self.queries[3]), -1)
        self.assertEqual(self.knn[2].acccuracy(self.knn[2].confusion_matrix(self.queries, self.test_y)), 1.0)


    def test3(self):
        self.assertEqual(self.knn[3].classify(self.queries[0]), 1)
        self.assertEqual(self.knn[3].classify(self.queries[1]), 1)
        self.assertEqual(self.knn[3].classify(self.queries[2]), 1)
        self.assertEqual(self.knn[3].classify(self.queries[3]), -1)
        self.assertEqual(self.knn[3].acccuracy(self.knn[3].confusion_matrix(self.queries, self.test_y)), 0.75)

if __name__ == '__main__':
    unittest.main()
