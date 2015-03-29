__author__ = 'carlxie'

import numpy as np
from BPNetwork import BPNetwork

shapes = [(2,3),(3,1)]
def compute_num_grad(net, x, y):
    e = 1e-4
    ls1 = flat_list(net.weights)
    ls2 = flat_list(net.biases)
    weights = np.append(ls1,ls2)

    perturb = np.zeros(weights.shape)
    num_grad = np.zeros(weights.shape)
    x.shape = (1,x.shape[0])
    for i in range(len(weights)):
        perturb[i] = e
        w1,b1 = reconstruct(weights + perturb, shapes)
        w2,b2 = reconstruct(weights - perturb, shapes)
        perturb[i] = 0
        s1 = net.cost(x,y,w1,b1)
        s2 = net.cost(x,y,w2,b2)
        num_grad[i] = (s1 - s2) / (2 * e)
    return num_grad


def flat_list(ls):
    fls = np.array([])
    for i in range(len(ls)):
        s = ls[i][:].shape
        r = ls[i][:].reshape(s[0] * s[1])
        fls = np.append(fls, r)
    return fls


def reconstruct(ls, shapes):
    length = 0
    rls = []
    bls = []
    for s in shapes:
        lls = ls[length:(length + s[0] * s[1])]
        length += s[0] * s[1]
        rls.append(lls.reshape(s[0], s[1]))
    for s in shapes:
        lls = ls[length:(length + s[1])]
        length += s[1]
        bls.append(lls.reshape(1,s[1]))
    return rls,bls


if __name__ == "__main__":
    train = np.loadtxt('hw4_nnet_train.dat')
    x, y = train[:, :-1], train[:, -1]
    net = BPNetwork([2, 3, 1],0.5)
    print compute_num_grad(net, x[1], y[1])
    w_grads,b_grads = net.calculate_analytic_grads(x[1], y[1])
    print np.append(flat_list(w_grads),flat_list(b_grads))
