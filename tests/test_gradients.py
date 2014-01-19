import unittest
import cPickle

from numpy import array, zeros, sqrt, finfo
from numpy.testing import assert_array_almost_equal
from scipy.optimize import check_grad, approx_fprime, minimize

from dataset import Dataset
from trainer import Trainer
from utils import synthetic
from models import Linear

class TestGradients(unittest.TestCase):

    def setUp(self):
        length = 100

        with open('tests/fixtures.pkl', 'rb') as pkl:
            self.prices_jnj = cPickle.load(pkl)[:length]
            self.prices_apl = cPickle.load(pkl)[:length]

        self.data = Dataset(self.prices_jnj, [self.prices_apl])
        self.trX, self.trY, _, _ = self.data.build(0, 100, 5, 50)

    def test_linear_mean_return_model(self):
        model = Linear(lookback=10, delta=0.1, lmb=1.)

        for i in range(10):
            diff = check_grad(model.cost,
                              model.grad,
                              model.weights(self.trX, i),
                              self.trX,
                              self.trY)

            self.assertTrue(diff < 1.e-8, diff)
