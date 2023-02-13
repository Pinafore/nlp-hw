import unittest

from lr_sgd_qb import LogReg, Example

kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example({"label":True, "A":4, "B":3, "C":1}, kTOY_VOCAB)
kNEG = Example({"label":False, "B":1, "C":3, "D":4}, kTOY_VOCAB)

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.logreg_unreg = LogReg(5, 1.0)

    def test_unreg(self):
        print(self.logreg_unreg.beta)
        print(kPOS.x)
        beta = self.logreg_unreg.sg_update(kPOS)
        self.assertAlmostEqual(beta[0], .5)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 1.5)
        self.assertAlmostEqual(beta[3], 0.5)
        self.assertAlmostEqual(beta[4], 0.0)

        print(self.logreg_unreg.beta)
        print(kPOS.x)
        beta = self.logreg_unreg.sg_update(kNEG)
        self.assertAlmostEqual(beta[0], -0.47068776924864364)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 0.5293122307513564)
        self.assertAlmostEqual(beta[3], -2.4120633077459308)
        self.assertAlmostEqual(beta[4], -3.8827510769945746)

if __name__ == '__main__':
    unittest.main()
