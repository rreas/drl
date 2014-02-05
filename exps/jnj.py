from sys import path
path.append('src/')

import cPickle

from dataset import Dataset
from models import Linear, Nonlinear
from trainer import Trainer

with open('tests/fixtures.pkl', 'rb') as pkl:
    prices_jnj = cPickle.load(pkl)
    prices_apl = cPickle.load(pkl)
    
    data = Dataset(prices_jnj, [])
    model = Nonlinear(delta=0.01, hidden=3)
    #model = Linear(delta=0.01)
    trainer = Trainer(data, model)

    trainer.train(1000, 20, 100)
