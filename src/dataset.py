from numpy import array, log
from series import Series
from window import Window

class Dataset:
    
    def __init__(self, raw):
        self.series = Series(raw)

    def make(self, data, lookback):
        x = []; y = []
        for i in range(0, len(data)-lookback):
            j = i + lookback
            x.append(data[i:j])
            y.append(data[j])
        return x, y

    def build(self, start, window, lookback, slide):
        trD = self.series.data[start:start+window]
        tsD = self.series.data[start+window-lookback:start+window+slide]
        trX, trY = self.make(trD, lookback)
        tsX, tsY = self.make(tsD, lookback)

        return trX, trY, tsX, tsY

    def gen(self, window=100, lookback=10, slide=50):
        start = 0

        while(start+window < len(self.series.data)):
            trX, trY, tsX, tsY = self.build(start, window, lookback, slide)
            yield trX, trY, tsX, tsY
            start += slide
