import unittest

from dataset import Dataset
from numpy import array, log, append
from numpy.testing import assert_array_almost_equal

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.side = [ [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ]
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
        self.generate(5, 3, 4)

        self.assertEqual(2, len(self.train_examples[0]))

        assert_array_almost_equal(self.train_examples[0][0],
                append([1./1, 1./2], log([2./1, 3./2, 4./3, 3./1, 4./2, 5./3])))
        assert_array_almost_equal(self.train_examples[0][1],
                append([1./2, 1./3], log([3./2, 4./3, 5./4, 4./2, 5./3, 6./4])))

        self.assertEqual(2, len(self.train_examples[1]))

        assert_array_almost_equal(self.train_examples[1][0],
                append([1./5, 1./6], log([6./5, 7./6, 8./7, 7./5, 8./6, 9./7])))
        assert_array_almost_equal(self.train_examples[1][1],
                append([1./6, 1./7], log([7./6, 8./7, 9./8, 8./6, 9./7, 10./8])))

    def test_creates_training_returns(self):
        self.generate(5, 3, 4)

        self.assertEqual(2, len(self.train_returns[0]))
        assert_array_almost_equal(self.train_returns[0][0], 1./3)
        assert_array_almost_equal(self.train_returns[0][1], 1./4)

        self.assertEqual(2, len(self.train_returns[1]))
        assert_array_almost_equal(self.train_returns[1][0], 1./7)
        assert_array_almost_equal(self.train_returns[1][1], 1./8)

    def test_creates_testing_examples(self):
        self.generate(5, 3, 4)

        self.assertEqual(4, len(self.test_examples[0]))

        assert_array_almost_equal(self.test_examples[0][0],
                append([1./3, 1./4], log([4./3, 5./4, 6./5, 5./3, 6./4, 7./5])))
        assert_array_almost_equal(self.test_examples[0][1],
                append([1./4, 1./5], log([5./4, 6./5, 7./6, 6./4, 7./5, 8./6])))
        assert_array_almost_equal(self.test_examples[0][2],
                append([1./5, 1./6], log([6./5, 7./6, 8./7, 7./5, 8./6, 9./7])))
        assert_array_almost_equal(self.test_examples[0][3],
                append([1./6, 1./7], log([7./6, 8./7, 9./8, 8./6, 9./7, 10./8])))

        self.assertEqual(1, len(self.test_examples[1]))

        assert_array_almost_equal(self.test_examples[1][0],
                append([1./7, 1./8], log([8./7, 9./8, 10./9, 9./7, 10./8, 11./9])))

    def test_creates_testing_returns(self):
        self.generate(5, 3, 4)

        self.assertEqual(4, len(self.test_returns[0]))
        assert_array_almost_equal(self.test_returns[0][0], 1./5)
        assert_array_almost_equal(self.test_returns[0][1], 1./6)
        assert_array_almost_equal(self.test_returns[0][2], 1./7)
        assert_array_almost_equal(self.test_returns[0][3], 1./8)

        self.assertEqual(1, len(self.test_returns[1]))
        assert_array_almost_equal(self.test_returns[1][0], 1./9)

    def test_generates_with_no_ending_test_examples(self):
        self.generate(5, 3, 5)
        self.assertEqual(1, len(self.train_returns))
        self.assertEqual(1, len(self.test_returns))
    
