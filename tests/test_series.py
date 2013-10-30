import unittest

from series import Series
from numpy import array, log
from numpy.testing import assert_array_almost_equal

class TestSeries(unittest.TestCase):

    def setUp(self):
        self.raw = array([1, 2, 3, 4])
        self.s = Series(self.raw)
        self.d = array([2./1, 3./2, 4./3])

    def test_uses_ratios(self):
        assert_array_almost_equal(self.s.data, self.d)
