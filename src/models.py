from numpy import tanh, append, dot, reshape, absolute, zeros, float64, mean
from numpy.random import RandomState
from numpy.linalg import norm

from utils import sech2, sign

class Model:

    def calc_r(self, d_t, d_prev, q):
        return d_t*q - self.delta*absolute(d_t - d_prev)

# FIXME: No bias term yet, no regularization, scaling of example vectors.
class Linear(Model):

    def __init__(self, lookback=10, delta=0.01, lmb=1.):
        self.lookback = lookback
        self.delta = delta
        self.lmb = lmb

    def deflate(self, w, alpha):
        return append(w, alpha)

    def inflate(self, w):
        return w[:-1], w[-1]

    def weights(self, x, seed=1):
        return RandomState(seed).randn(len(x[0])+1)

    def decide(self, d_prev, x_t, w, alpha):
        return tanh(dot(w, x_t) + alpha*d_prev)

    def mean_return(self, params, x, y):
        w, alpha = self.inflate(params)
        total = d_t = d_prev = 0.

        for i, x_t in enumerate(x):
            d_t = self.decide(d_prev, x_t, w, alpha)
            total += self.calc_r(d_t, d_prev, y[i])
            d_prev = d_t

        return total / len(y)

    def cost(self, params, x, y):
        return -1 * self.mean_return(params, x, y)

    def grad(self, params, x, y):
        w, alpha = self.inflate(params)

        d_prev = self.decide(0., x[0], w, alpha)

        d_prev_by_w = sech2(dot(w, x[0])) * x[0]
        r_prev_by_w = d_prev_by_w*y[0] - self.delta*sign(d_prev)*d_prev_by_w
        A_prev_by_w = r_prev_by_w

        d_prev_by_alpha = r_prev_by_alpha = A_prev_by_alpha = 0.

        for i in range(1, len(y)):
            x_t = x[i]
            q_t = y[i]

            d_t = self.decide(d_prev, x_t, w, alpha)
            d_i = dot(w, x_t) + alpha*d_prev

            d_t_by_w = sech2(d_i) * (x_t + alpha*d_prev_by_w)
            r_t_by_w = d_t_by_w*q_t - self.delta*sign(d_t - d_prev)*(d_t_by_w - d_prev_by_w)
            A_t_by_w = A_prev_by_w + (1./(i+1))*(r_t_by_w - A_prev_by_w)

            if i == 1:
                d_t_by_alpha = sech2(d_i) * d_prev
            else:
                d_t_by_alpha = sech2(d_i) * (alpha * d_prev_by_alpha + d_prev)

            r_t_by_alpha = d_t_by_alpha*q_t - (
                    self.delta*sign(d_t - d_prev)*(d_t_by_alpha - d_prev_by_alpha))
            A_t_by_alpha = A_prev_by_alpha + (1./(i+1))*(r_t_by_alpha - A_prev_by_alpha)

            d_prev = d_t

            d_prev_by_w = d_t_by_w
            r_prev_by_w = r_t_by_w
            A_prev_by_w = A_t_by_w

            d_prev_by_alpha = d_t_by_alpha
            r_prev_by_alpha = r_t_by_alpha
            A_prev_by_alpha = A_t_by_alpha

        return -1 * append(A_t_by_w, A_t_by_alpha)
