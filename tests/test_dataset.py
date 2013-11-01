import unittest

from dataset import Dataset
from numpy import array, log
from numpy.testing import assert_array_almost_equal

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.raw = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.d = Dataset(self.raw)

        window = 5
        lookback = 2
        slide = 3

        self.train_examples = []
        self.train_returns = []
        self.test_examples = []
        self.test_returns = []

        for trX, trY, tsX, tsY in self.d.gen(window=5, lookback=2, slide=3):
            self.train_examples.append(trX)
            self.train_returns.append(trY)
            self.test_examples.append(tsX)
            self.test_returns.append(tsY)

    def test_creates_training_examples(self):
        assert_array_almost_equal([[[2./1, 3./2], [3./2, 4./3], [4./3, 5./4]],
                                   [[5./4, 6./5], [6./5, 7./6], [7./6, 8./7]]],
                                  self.train_examples)

    def test_creates_training_returns(self):
        assert_array_almost_equal([[4./3, 5./4, 6./5],
                                   [7./6, 8./7, 9./8]],
                                  self.train_returns)

    def test_creates_testing_examples(self):
        assert_array_almost_equal([[5./4, 6./5], [6./5, 7./6], [7./6, 8./7]],
                                  self.test_examples[0])
        assert_array_almost_equal([[8./7, 9./8]], 
                                  self.test_examples[1])

    def test_creates_testing_returns(self):
        assert_array_almost_equal([7./6, 8./7, 9./8], self.test_returns[0])
        assert_array_almost_equal([10./9,], self.test_returns[1])
