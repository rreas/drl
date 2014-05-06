import unittest

from utils import sharpe
from numpy import array

class TestSharpe(unittest.TestCase):

    def setUp(self):
        self.returns = array([1, 2, -1, 3, 2])

    def test_calculates_sharpe(self):
        self.assertAlmostEqual(sharpe(self.returns)[-1], 1.03209369)
