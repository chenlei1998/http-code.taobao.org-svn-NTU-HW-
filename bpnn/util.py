__author__ = 'carlxie'

import numpy as np

# remember our old friend ??
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def uniform_rand(r, shape):
    return 2 * r * np.random.random(shape) - r

## tanh is just a scaled and shifted sigmoid function
def tanh(s):
    return 2 * sigmoid(2 * s) - 1


def der_tanh(s):
    t = sigmoid(2 * s)
    return 4 * t * (1 - t)

def rand_pick(X):
    index = int(np.random.random() * len(X))
    return X[index][:-1], X[index][-1]

# vectorize our util function
vec_tanh = np.vectorize(tanh)
vec_der_tanh = np.vectorize(der_tanh)

def sign(v):
    if v > 0:return 1
    else:return -1