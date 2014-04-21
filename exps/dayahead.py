from sys import path
path.append('src/')

import cPickle
from numpy import zeros
import matplotlib.pyplot as plt

from dataset import Dataset
from models import Linear, Nonlinear
from trainer import Trainer

from quote import get

with open('data/jnj_yahoo.pkl', 'rb') as pkl:
    jnj = cPickle.load(pkl)

jnj_curr = jnj[0:1000]
jnj_next = jnj[1:1001]

data = Dataset(jnj_curr, [jnj_next])
#model = Nonlinear(delta=0.01, hidden=3)
model = Linear(delta=0.01)
trainer = Trainer(data, model)

_, trR, tsR = trainer.train(200, 5, 100, maxiter=500)

x = range(len(trR))
plt.plot(x, trR, 'r', x, tsR, 'b')
plt.show()
