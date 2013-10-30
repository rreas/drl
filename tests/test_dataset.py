import unittest

from dataset import Dataset
from numpy import array, log
from numpy.testing import assert_array_almost_equal

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.raw = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.d = Dataset(self.raw)
        self.x = []
        self.y = []

        for x, y in self.d.gen(size=5, lookback=2):
            self.x.append(x)
            self.y.append(y)

    def test_creates_examples(self):
        assert_array_almost_equal([[[2./1, 3./2], [3./2, 4./3], [4./3, 5./4]],
                                   [[3./2, 4./3], [4./3, 5./4], [5./4, 6./5]],
                                   [[4./3, 5./4], [5./4, 6./5], [6./5, 7./6]],
                                   [[5./4, 6./5], [6./5, 7./6], [7./6, 8./7]],
                                   [[6./5, 7./6], [7./6, 8./7], [8./7, 9./8]]],
                                  self.x)

    def test_creates_returns(self):
        assert_array_almost_equal(log([[4./3, 5./4, 6./5],
                                       [5./4, 6./5, 7./6],
                                       [6./5, 7./6, 8./7],
                                       [7./6, 8./7, 9./8],
                                       [8./7, 9./8, 10./9]]),
                                  self.y)
