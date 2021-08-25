import unittest
import tempfile
import torch

from torch.nn import BCELoss as Loss
from torch.optim import SGD as Opt

from lr_pytorch import step, SimpleLogreg
from numpy import array

class TestPyTorchLR(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        self.raw_data = array([[1., 4., 3., 1., 0.],
                              [0., 0., 1., 3., 4.]])
        self.data = torch.from_numpy(self.raw_data).float()

        labels = array([[1], [0]])
        self.labels = torch.from_numpy(labels).float()
        
        self.model = SimpleLogreg(5)

        with torch.no_grad():
            self.model.linear.weight.fill_(0)
            self.model.linear.weight[0,0] = 1
            self.model.linear.weight[0,4] = -1        
 
    def test_forward(self):
        self.assertAlmostEqual(0.667, float(self.model.forward(self.data)[0]), 3)
        self.assertAlmostEqual(0.0135, float(self.model.forward(self.data)[1]), 3)

    def test_step(self):
        optimizer = Opt(self.model.parameters(), lr=0.1)
        criterion = Loss()
        step(0, 0, self.model, optimizer, criterion, self.data, self.labels)

        weight, bias = list(self.model.parameters())
        self.assertAlmostEqual(float(weight[0][0]), 1.0166, 3)
        self.assertAlmostEqual(float(weight[0][1]), 0.0666, 3)
        self.assertAlmostEqual(float(weight[0][2]), 0.0493, 3)
        self.assertAlmostEqual(float(weight[0][3]), 0.0145, 3)
        self.assertAlmostEqual(float(weight[0][4]), -1.0027, 3)

        self.assertAlmostEqual(float(bias[0]), -0.289, 3)
        
if __name__ == '__main__':
    unittest.main()
