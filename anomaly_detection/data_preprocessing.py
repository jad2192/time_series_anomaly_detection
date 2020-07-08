import numpy as np


def get_counts(rows, dates, brands):
    res = {}
    for brand in brands:
        res[brand] = np.zeros(len(dates))
    for row in rows:
        res[row[1]][dates.index(row[2])] = row[0]
    return res


def discretize(counts, cnt='day'):
    mu = np.median(counts)
    sigs = []
    if cnt == 'day':
        L = 7
    else:
        L = 4
    for k in range(L, len(counts), L):
        sigs.append(counts[k:k + L].std())
    sig = np.mean(sigs)
    if sig == 0:
        sig = 1
    res = []
    for x in counts:
        if x - mu <= sig:
            res.append(0)
        elif (sig < x - mu) and (x - mu <= 2 * sig):
            res.append(1)
        else:
            res.append(2)
    return np.asanyarray(res, dtype='i4')


def discretize_dec(counts, cnt='day'):
    mu = np.median(counts)
    sigs = []
    if cnt == 'day':
        L = 7
    else:
        L = 4
    for k in range(L, len(counts), L):
        sigs.append(counts[k:k + L].std())
    sig = np.mean(sigs)
    if sig == 0:
        sig = 1
    res = []
    for x in counts:
        if x - mu >= -sig:
            res.append(0)
        elif (-sig > x - mu) and (x - mu >= -2 * sig):
            res.append(1)
        else:
            res.append(2)
    return np.asanyarray(res, dtype='i4')

