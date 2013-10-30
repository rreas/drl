from numpy import array, log
from series import Series
from window import Window

class Dataset:
    
    def __init__(self, raw):
        self.raw = raw

    def gen(self, size=100, lookback=10):
        for data in Window(self.raw, size=size).gen():
            x = []
            y = []

            for i in range(0, len(data)-lookback):
                j = i + lookback
                x.append(data[i:j])
                y.append(data[j])

            # Yields all examples for the current window.
            yield x, log(y)
