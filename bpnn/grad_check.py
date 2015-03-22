import numpy as np
from Network import Network

###
## first compute numerical gradient of
## each weight in the net:
## dw = f(x+h)-f(x-h)/2h
###

shapes = [(3, 3), (4, 1)]


def compute_num_grad(net, x, y):
    e = 1e-4
    weights = flat_list(net.W)
    perturb = np.zeros(weights.shape)
    num_grad = np.zeros(weights.shape)
    for i in range(len(weights)):
        perturb[i] = e
        w1 = reconstruct(weights + perturb, shapes)
        w2 = reconstruct(weights - perturb, shapes)
        perturb[i] = 0
        s1 = net.cost(x, y, w1)
        s2 = net.cost(x, y, w2)
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
    for s in shapes:
        lls = ls[length:(length + s[0] * s[1])]
        length += s[0] * s[1]
        rls.append(lls.reshape(s[0], s[1]))
    return rls


if __name__ == "__main__":
    train = np.loadtxt('hw4_nnet_train.dat')
    x, y = train[:, :-1], train[:, -1]
    net = Network(30000, [2, 3, 1], 0.01, 0.5,10,0.3)
    print compute_num_grad(net, x[0], y[0])
    print flat_list(net.calculate_analytic_grads(x[0], y[0]))
