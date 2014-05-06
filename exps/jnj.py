from sys import path
path.append('src/')

import cPickle
from numpy import zeros, append
import matplotlib.pyplot as plt

from dataset import Dataset
from models import Linear, Nonlinear
from trainer import AdaGradTrainer
from utils import synthetic, wealth, sharpe

from quote import get

FETCH = False

if FETCH:
    start = '2007-07-01'
    finish = '2014-02-01'
    
    jnj_data = get('JNJ', start, finish)
    cov_data = get('COV', start, finish)
    nvs_data = get('NVS', start, finish)
    pfe_data = get('PFE', start, finish)
    
    days = len(jnj_data)
    jnj = zeros(days)
    cov = zeros(days)
    nvs = zeros(days)
    pfe = zeros(days)
    
    for i in range(days):
        jnj[i] = jnj_data[i][1]['c']
        cov[i] = cov_data[i][1]['c']
        nvs[i] = nvs_data[i][1]['c']
        pfe[i] = pfe_data[i][1]['c']
    
    with open('data/jnj_yahoo.pkl', 'wb') as pkl:
        cPickle.dump(jnj, pkl)
        cPickle.dump(cov, pkl)
        cPickle.dump(nvs, pkl)
        cPickle.dump(pfe, pkl)

else:
    with open('data/jnj_yahoo.pkl', 'rb') as pkl:
        jnj = cPickle.load(pkl)
        cov = cPickle.load(pkl)
        nvs = cPickle.load(pkl)
        pfe = cPickle.load(pkl)

window = 100
slide = 50
lookback = 5
delta = 0.0005
lmb = 0.1

hidden = 10

model = Linear(delta=delta, lmb=lmb)
#model = Nonlinear(delta=delta, lmb=lmb, hidden=hidden)

data = Dataset(cov, [jnj, nvs, pfe])
trainer = AdaGradTrainer(data, model)

returns, decisions = trainer.train(
        window=window, lookback=lookback, slide=slide, maxiter=40)
padding = zeros(len(jnj)-len(returns))
returns = append(padding, returns)
decisions = append(padding, decisions)

print "Wealth: ", wealth(returns)[-1]

x_axis = range(len(jnj))

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x_axis, jnj / jnj[0], 'r', label='Prices')
axarr[0].plot(x_axis, wealth(returns), 'b', label='Wealth')
axarr[0].set_title('Performance')
axarr[0].legend(loc="upper left")

#axarr[1].plot(x_axis, sharpe(returns))
#axarr[1].set_title('Sharpe')
#axarr[3].plot(x_axis, returns)
#axarr[3].set_title('Returns')
axarr[1].plot(x_axis, decisions)
axarr[1].set_title('Decisions')
plt.show()

