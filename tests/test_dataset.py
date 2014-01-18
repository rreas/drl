import unittest

from dataset import Dataset
from numpy import array, log
from numpy.testing import assert_array_almost_equal

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.raw = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.side = array([ [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ])
        self.d = Dataset(self.raw, self.side)

    def generate(self, window, lookback, slide):
        self.train_examples = []
        self.train_returns = []
        self.test_examples = []
        self.test_returns = []

        for trX, trY, tsX, tsY in self.d.gen(window=window,
                                             lookback=lookback,
                                             slide=slide):
            self.train_examples.append(trX)
            self.train_returns.append(trY)
            self.test_examples.append(tsX)
            self.test_returns.append(tsY)

    def test_creates_training_examples(self):
        self.generate(5, 2, 3)
        assert_array_almost_equal([[[2./1, 3./2], [3./2, 4./3], [4./3, 5./4]],
                                   [[5./4, 6./5], [6./5, 7./6], [7./6, 8./7]]],
                                  self.train_examples)

    def test_creates_training_returns(self):
        self.generate(5, 2, 3)
        assert_array_almost_equal([[4./3, 5./4, 6./5],
                                   [7./6, 8./7, 9./8]],
                                  self.train_returns)

    def test_creates_testing_examples(self):
        self.generate(5, 2, 3)
        assert_array_almost_equal([[5./4, 6./5], [6./5, 7./6], [7./6, 8./7]],
                                  self.test_examples[0])
        assert_array_almost_equal([[8./7, 9./8]], 
                                  self.test_examples[1])

    def test_creates_testing_returns(self):
        self.generate(5, 2, 3)
        assert_array_almost_equal([7./6, 8./7, 9./8], self.test_returns[0])
        assert_array_almost_equal([10./9,], self.test_returns[1])

    def test_creates_one_test_set_with_long_slide(self):
        self.generate(5, 2, 4)
        assert_array_almost_equal([7./6, 8./7, 9./8, 10./9], self.test_returns[0])

