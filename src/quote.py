import urllib
from datetime import datetime

def get(sym, start, end):
    """Gets symbol data for a given date range and symbol"""
    fmt = '%Y-%m-%d'
    s = datetime.strptime(start, fmt)
    f = datetime.strptime(end, fmt)

    url = ['http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv',
           '&s=%s' % sym,
           '&a=%i' % (s.month-1),
           '&b=%i' % s.day,
           '&c=%i' % s.year,
           '&d=%i' % (f.month-1),
           '&e=%i' % f.day,
           '&f=%i' % f.year]
    url = ''.join(url)
    return build_data_list(urllib.urlopen(url).readlines())

def day(sym, date):
    """Gets symbol data for a single day."""
    return get(sym, date, date)[0][1]

def build_data_list(data):
    res = []
    for i in xrange(1, len(data)):
        row = data[i].split(',')
        hsh = {'o': float(row[1]), # open
               'h': float(row[2]), # high
               'l': float(row[3]), # low
               'c': float(row[4])} # close
        res.append((row[0], hsh))

    # Put oldest at start of list.
    res.reverse()
    return res

