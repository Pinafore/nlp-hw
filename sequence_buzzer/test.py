import unittest
from buzzer import QuestionDataset, RNNBuzzer, create_feature_vecs_and_labels
import numpy as np
import torch
import torch.nn as nn

torch.set_printoptions(precision=10)


ex1 = {'feature_vec':torch.FloatTensor([[[0.1334, 0.1011, 0.0932], [0.1501, 0.1001, 0.0856], [0.1647, 0.0987, 0.0654]]]).view(1, 3, 3), 'len': torch.FloatTensor([3])}

ex2 = {'feature_vec':torch.FloatTensor([[[0.1234, 0.1111, 0.0934], [0.1301, 0.1041, 0.0898], [0.1447, 0.0981, 0.0723], [0.1596, 0.0901, 0.0657]],
									   [[0.1034, 0.0983, 0.0679], [0.1555, 0.1144, 0.0882], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
									   [[0.1132, 0.0932, 0.0813], [0.1404, 0.1001, 0.0831], [0.1696, 0.0777, 0.0593], [0.0, 0.0, 0.0]]]),
										'len': torch.FloatTensor([4, 2, 3])}



class TestSeq2Seq(unittest.TestCase):

	def setUp(self):
		self.toy_rnn_model = RNNBuzzer(n_input=3, n_hidden=2)
		self.toy_rnn_model.eval()

		lstm_weight_input_hidden = torch.tensor([[0.4, -0.2, 0.1],
		 										 [-0.4, 0.5, 0.2],
		 										 [0.3, 0.2, 0.1],
		 										 [0.4, 0.9, -0.1],
		 										 [0.8, -0.7, -0.5],
		 										 [0.7, 0.1, -0.1],
		 										 [0.0, 0.1, 0.0],
		 										 [-0.9, -0.8, -0.7]])
		lstm_weight_hidden_hidden = torch.tensor([[0.5, -0.1],
		 										 [-0.4, 0.3],
		 										 [0.3, 0.6],
		 										 [0.4, -0.2],
		 										 [0.8, -0.9],
		 										 [-0.7, 0.0],
		 										 [0.5, 0.2],
		 										 [0.0, -0.5]])

		self.toy_rnn_model.lstm.weight_ih_l0.data.copy_(lstm_weight_input_hidden)
		self.toy_rnn_model.lstm.weight_hh_l0.data.copy_(lstm_weight_hidden_hidden)
		self.toy_rnn_model.lstm.bias_ih_l0.data.fill_(1.0)
		self.toy_rnn_model.lstm.bias_hh_l0.data.fill_(1.0)

		hidden_linear_layer_weight = torch.tensor([[0.4, -0.2], [-0.9, 0.8]])
		self.toy_rnn_model.hidden_to_label.weight.data.copy_(hidden_linear_layer_weight)

		nn.init.ones_(self.toy_rnn_model.hidden_to_label.bias.data)

	def test_forward(self):
		logits = self.toy_rnn_model(ex1['feature_vec'], ex1['len'])

		self.assertAlmostEqual(logits[0][0].item(), 1.126254796981)
		self.assertAlmostEqual(logits[0][1].item(), 0.922435641288757)
		self.assertAlmostEqual(logits[1][0].item(), 1.193930149078369)
		self.assertAlmostEqual(logits[1][1].item(), 0.8235720992088)
		self.assertAlmostEqual(logits[2][0].item(), 1.2111276388168)
		self.assertAlmostEqual(logits[2][1].item(), 0.796994566917)


	def test_minibatch(self):
		logits = self.toy_rnn_model(ex2['feature_vec'], ex2['len'])

		self.assertAlmostEqual(logits[0][0].item(), 1.1259287596)
		self.assertAlmostEqual(logits[0][1].item(), 0.9232868552)
		self.assertAlmostEqual(logits[1][0].item(), 1.1934133768)
		self.assertAlmostEqual(logits[1][1].item(), 0.8253083229)
		self.assertAlmostEqual(logits[2][0].item(), 1.2106758356)
		self.assertAlmostEqual(logits[2][1].item(), 0.7986904979)
		self.assertAlmostEqual(logits[3][0].item(), 1.214038729)
		self.assertAlmostEqual(logits[3][1].item(), 0.7943208218)
		self.assertAlmostEqual(logits[4][0].item(), 1.1251035929)
		self.assertAlmostEqual(logits[4][1].item(), 0.9263896942)
		self.assertAlmostEqual(logits[5][0].item(), 1.1943942308)
		self.assertAlmostEqual(logits[5][1].item(), 0.8215977550)
		self.assertAlmostEqual(logits[6][0].item(), 1.2029464245)
		self.assertAlmostEqual(logits[6][1].item(), 0.8289564848)
		self.assertAlmostEqual(logits[7][0].item(), 1.2067799568)
		self.assertAlmostEqual(logits[7][1].item(), 0.8231585622)
		self.assertAlmostEqual(logits[8][0].item(), 1.1255118847)
		self.assertAlmostEqual(logits[8][1].item(), 0.9250283241)
		self.assertAlmostEqual(logits[9][0].item(), 1.1935989857)
		self.assertAlmostEqual(logits[9][1].item(), 0.8247293830)
		self.assertAlmostEqual(logits[10][0].item(), 1.2105900049)
		self.assertAlmostEqual(logits[10][1].item(), 0.7990440130)
		self.assertAlmostEqual(logits[11][0].item(), 1.2060568333)
		self.assertAlmostEqual(logits[11][1].item(), 0.8258087635)
		



	def test_feature_and_label_vectorizer(self):
		guesses_and_scores1 = [[[('Little_Brown_Foxes', 0.1435), ('Jerry_Seinfeld', 0.1332), ('India', 0.1198)], 
		[('United_States', 0.1335), ('England', 0.1212), ('Canada', 0.1011)],
		[('England', 0.1634), ('United_States', 0.1031), ('France', 0.0821)]]]

		ans1 = [['England', 'England', 'England']]
		exs = create_feature_vecs_and_labels(guesses_and_scores1, ans1, 3)

		self.assertEqual(exs[0][0][0][0], 0.1435)
		self.assertEqual(exs[0][0][0][1], 0.1332)
		self.assertEqual(exs[0][0][1][1], 0.1212)
		self.assertEqual(exs[0][0][1][2], 0.1011)
		self.assertEqual(exs[0][0][2][0], 0.1634)
		self.assertEqual(exs[0][0][2][2], 0.0821)

		self.assertEqual(exs[0][1][0], 0)
		self.assertEqual(exs[0][1][1], 0)
		self.assertEqual(exs[0][1][2], 1)

        



if __name__ == '__main__':
    unittest.main()
