from sys import path
path.append('src/')

import cPickle
from numpy import zeros
import matplotlib.pyplot as plt

from dataset import Dataset
from models import Linear, Nonlinear
from trainer import Trainer

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

data = Dataset(cov, [jnj, nvs, pfe])
#model = Nonlinear(delta=0.01, hidden=3)
model = Linear(delta=0.01)
trainer = Trainer(data, model)

_, trR, tsR = trainer.train(1000, 5, 100, maxiter=500)

x = range(len(trR))
plt.plot(x, trR, 'r', x, tsR, 'b')
plt.show()
