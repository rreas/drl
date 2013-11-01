from sys import path
path.append('src/')

from network import Network

with(open('data/jnj.csv', 'r')

network = Network([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=5, lookback=2)
network.train()
