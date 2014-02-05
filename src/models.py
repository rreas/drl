from numpy import (tanh,
                   concatenate,
                   append,
                   dot,
                   reshape,
                   absolute,
                   zeros,
                   float64,
                   mean,
                   outer,
                   diag,
                   eye)

from numpy.random import RandomState
from numpy.linalg import norm

from utils import sech2, sign

class Model:

    def calc_r(self, d_t, d_prev, q):
        return d_t*q - self.delta*absolute(d_t - d_prev)

    def __init__(self, delta=0.01, lmb=1.):
        self.delta = delta
        self.lmb = lmb

    def cost(self, params, x, y):
        return -1 * self.mean_return(params, x, y)

class Nonlinear(Model):

    def __init__(self, delta=0.01, lmb=1., hidden=10):
        Model.__init__(self, delta, lmb)
        self.hidden = hidden

    def weights(self, data, seed=1):
        self.exlen = len(data[0])
        return RandomState(seed).randn(self.exlen*self.hidden + 2*self.hidden)

    def deflate(self, W, w, alpha):
        append(W.reshape((W.size,)), w, alpha)

    def inflate(self, params):
        a = self.exlen * self.hidden
        b = a + self.hidden

        W = params[:a].reshape((self.hidden, self.exlen))
        w = params[a:b]
        alpha = params[b:]

        return W, w, alpha

    def decide(self, d_prev, x_t, W, w, alpha):
        return tanh(dot(w, tanh(dot(W, x_t) + alpha*d_prev)))

    def grad(self, params, x, y):
        W, w, alpha = self.inflate(params)

        x_t = x[0]
        q_t = y[0]
        h_t = tanh(dot(W, x_t))
        
        d_prev = self.decide(0., x_t, W, w, alpha)

        d_prev_by_w = sech2(dot(w, h_t)) * h_t
        r_prev_by_w = d_prev_by_w*y[0] - self.delta*sign(d_prev)*d_prev_by_w
        A_prev_by_w = r_prev_by_w

        h_prev_by_W = zeros((w.size, W.shape[0], W.shape[1]))
        for i in range(self.hidden):
            h_prev_by_W[i,:,:] = sech2(dot(W[i,:], x_t)) * x_t

        d_prev_by_W = zeros(W.shape) + sech2(dot(w, h_t))
        for i in range(self.hidden):
            d_prev_by_W[i,:] *= w[i]*h_prev_by_W[i,i,:]

        r_prev_by_W = d_prev_by_W*y[0] - self.delta*sign(d_prev)*d_prev_by_W
        A_prev_by_W = r_prev_by_W

        d_prev_by_alpha = zeros(alpha.shape)
        r_prev_by_alpha = zeros(alpha.shape)
        A_prev_by_alpha = zeros(alpha.shape)

        for step in range(1, len(y)):
            x_t = x[step]
            q_t = y[step]
            h_t = tanh(dot(W, x_t) + alpha*d_prev)
            d_t = self.decide(d_prev, x_t, W, w, alpha)

            h_t_by_w = dot(outer(d_prev_by_w, alpha), diag(sech2(dot(W, x_t) + alpha*d_prev)))
            d_t_by_w = sech2(dot(w, h_t)) * (dot(h_t_by_w, w) + h_t)
            r_t_by_w = d_t_by_w*q_t - self.delta*sign(d_t - d_prev)*(d_t_by_w - d_prev_by_w)
            A_t_by_w = A_prev_by_w + (1./(step+1))*(r_t_by_w - A_prev_by_w)

            h_t_by_W = zeros((w.size, W.shape[0], W.shape[1]))
            for k in range(w.size):
                a = sech2(dot(W[k,:], x_t) + alpha[k]*d_prev)
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        b = alpha[k]*d_prev_by_W[i][j]
                        if k == i:
                            b += x_t[j]
                        h_t_by_W[k][i][j] = a*b

            d_t_by_W = zeros(W.shape)
            for i in range(self.hidden):
                for j in range(self.exlen):
                    d_t_by_W[i][j] = sech2(dot(w, h_t)) * dot(w, h_t_by_W[:,i,j])

            r_t_by_W = d_t_by_W*q_t - self.delta*sign(d_t - d_prev)*(d_t_by_W - d_prev_by_W)
            A_t_by_W = A_prev_by_W + (1./(step+1))*(r_t_by_W - A_prev_by_W)

            a = sech2(dot(W, x_t) + alpha*d_prev)
            b = outer(d_prev_by_alpha, alpha) + eye(alpha.size)*d_prev
            h_t_by_alpha = dot(b, diag(a))

            d_t_by_alpha = sech2(dot(w, h_t)) * dot(h_t_by_alpha, w)
            r_t_by_alpha = d_t_by_alpha*q_t - self.delta*sign(d_t - d_prev)*(d_t_by_alpha -
                                                                             d_prev_by_alpha)
            A_t_by_alpha = A_prev_by_alpha + (1./(step+1))*(r_t_by_alpha - A_prev_by_alpha)

            d_prev = d_t

            d_prev_by_w = d_t_by_w
            r_prev_by_w = r_t_by_w
            A_prev_by_w = A_t_by_w

            d_prev_by_W = d_t_by_W
            r_prev_by_W = r_t_by_W
            A_prev_by_W = A_t_by_W

            d_prev_by_alpha = d_t_by_alpha
            r_prev_by_alpha = r_t_by_alpha
            A_prev_by_alpha = A_t_by_alpha

        return -1 * concatenate((A_t_by_W.reshape((A_t_by_W.size,)), A_t_by_w, A_t_by_alpha))
    
# FIXME: refactor to dry
    def mean_return(self, params, x, y):
        W, w, alpha = self.inflate(params)
        total = d_t = d_prev = 0.

        for i, x_t in enumerate(x):
            d_t = self.decide(d_prev, x_t, W, w, alpha)
            total += self.calc_r(d_t, d_prev, y[i])
            d_prev = d_t

        return total / len(y)

# FIXME: no regularization, scaling of example vectors.
class Linear(Model):

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
