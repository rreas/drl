from numpy import array
from series import Series

class Window:
    
    def __init__(self, raw, size=100):
        self.series = Series(raw)
        self.data = self.series.data
        self.size = min(size, len(self.data))

    def data(self):
        return self.series.data[self.start:self.finish]

    def gen(self):
        start = 0
        while(start + self.size <= len(self.data)):
            yield self.data[start:start+self.size]
            start += 1
