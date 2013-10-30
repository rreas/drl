import unittest

from series import Series
from window import Window
from numpy import array, log
from numpy.testing import assert_array_almost_equal

class TestSeries(unittest.TestCase):

    def setUp(self):
        self.raw = array([1, 2, 3, 4, 5, 6, 7, 8])

    def get_data_from_windows(self):
        l = []
        for win in self.w.gen():
            l.append(win)
        return l

    def test_windows(self):
        self.w = Window(self.raw, size=5)
        assert_array_almost_equal([ [2./1, 3./2, 4./3, 5./4, 6./5],
                                    [3./2, 4./3, 5./4, 6./5, 7./6],
                                    [4./3, 5./4, 6./5, 7./6, 8./7] ],
                                  self.get_data_from_windows())

    def test_with_overflow(self):
        self.w = Window(self.raw, size=20)
        assert_array_almost_equal([ [2./1, 3./2, 4./3, 5./4, 6./5, 7./6, 8./7] ],
                                  self.get_data_from_windows())

