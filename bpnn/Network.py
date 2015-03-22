__author__ = 'carlxie'

import time
import numpy as np
from util import *
from multiprocessing import Process,Queue

class Network():
    ###
    ##  use SGD to train our network T times
    ##  for a 2 X 3 X 3 network,shape would be like [2,3,3]
    ##  eta is the learning rate
    ##  r just the range we use the sample the initial weights
    ###
    def __init__(self, T, shape, eta, r, mini_batch_size,lamb):
        self.T = T
        self.eta = eta
        self.shape = np.array(shape)
        # lambda hyper parameter that use to control the trade-off between data loss and regularization loss
        self.lamb = lamb
        self.mini_batch_size = mini_batch_size
        self.W = [uniform_rand(r, (shape[i] + 1, shape[i + 1])) for i in range(len(shape) - 1)]

    ###
    ##  note that all the neurons including the OUTPUT neurons in our network
    ##  used the tanh transformation
    ##  here we forward to compute the activation of the final layer
    ###
    def feedforward(self, x, weights):
        activation = x  # our input as the first activation
        for w in weights:
            activation = np.append(1, activation)  # add the bias term
            score = activation.dot(w)   # score = W * x ,remember?
            activation = vec_tanh(score) # use tanh to transform our score to be within (-1,1)
        return activation

    # we use the square error measure
    def cost(self, x, y, weights):
        return (y - self.feedforward(x, weights)) ** 2

    ###
    ##  for binary classification problem use the sign of the score of the final layer
    ##  for mult-ti class problem,use argmax(score) to predict instead
    ##  note: we use tanh on the last layer,so its score has been transformed
    ##        but it does not change the result because tanh is monotonically
    ###
    def predict(self, x, weights):
        return sign(self.feedforward(x, weights))

    ###
    ##  test function
    ###
    def evaluate(self, X, Y):
        return sum([self.predict(X[i], self.W) != Y[i]
                    for i in range(len(X))]) / float(len(X))

    ###
    ##  SGD is short for stochastic gradient descent
    ##  for any function f(x;w),x is the input,w is parameter:
    ##  1. compute the gradient delta (df/dw)
    ##  2. update the weight w = w - eta * delta
    ##     (eta is the learning rate)
    ###
    def SGD_train(self, training_data):
         for t in range(self.T):
            x, y = rand_pick(training_data)  # that is why it called stochastic
            ## all the dirty work get done here
            grads = self.calculate_analytic_grads(x,y)
            ## finally update all the weights
            self.update_weights(grads)


    def update_weights(self,grads):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.eta * grads[i]

    ###
    ##  no big change from SGD, just we partition X into small
    ##  patches which size is determined by mini_batch_size
    ##  and then use the average of those patches' gradients
    ##  to update our weights
    ###
    def mini_batch_SGD(self,training_data):
        for t in range(self.T):
            # first we random shuffle our data
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+self.mini_batch_size]
                    for k in xrange(0, len(training_data), self.mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

    def update_mini_batch(self,mini_batch):
        grads = [np.zeros(w.shape) for w in self.W]
        for data in mini_batch:
            cal_grad = self.calculate_analytic_grads(data[:-1],data[-1])
            for j in range(len(grads)):
                grads[j] = grads[j] + cal_grad[j]

        for j in range(len(grads)):
            grads[j] = grads[j] / self.mini_batch_size

        self.update_weights(grads)

    ## just for gradient check
    def calculate_analytic_grads(self, x, y):
        activations, scores = self.compute_activations_scores(x, y)

        ## backward pass, compute the delta of each neuron
        ## after the deltas are computed,use them to compute
        ## the gradient of each weight
        deltas = self.back_propagation(scores, x, y)

        ## grad = delta * a , remember?
        grads = self.compute_gradient(activations, deltas)
        return grads

    def compute_activations_scores(self, x, y):
        ## first we forward pass to compute and cache the
        ## score of each layer, in the meanwhile we store
        ## all the activation of each neuron for later use
        scores = []
        activations = []
        activation = x
        for w in self.W:
            activation = np.append(1, activation)  # add bias term
            activations.append(activation)
            score = activation.dot(w)
            scores.append(score)
            activation = vec_tanh(score)  # feature transforms
        return activations, scores

    def compute_gradient(self, activations, deltas):
        gradients = [np.empty(w.shape) for w in self.W]
        for i in range(len(self.W)):
            ## the origin shape of deltas[i] is like (n,),we have to
            ## force to make its shape like (n,1)
            deltas[i].shape = (deltas[i].shape[0], 1)
            activations[i].shape = (1, activations[i].shape[0])
            gradients[i] = deltas[i].dot(activations[i]).T
        return gradients

    def back_propagation(self, scores, x, y):
        # init all delta of each neuron including the bias neuron
        deltas = [None for size in self.shape[1:]]
        init_delta = self.get_last_layer_delta(scores[-1], y)
        deltas[-1] = np.append(1, init_delta)  # add the bias neuron delta for convenience

        ## backward compute the deltas,note the input layer is not included
        for layer in range(len(deltas) - 1, 0, -1):
            next_delta = deltas[layer][1:] # remove the bias neuron
            t = self.W[layer].dot(next_delta) # compute the weighted sum of next layer's deltas
            m = vec_der_tanh(np.append(1, scores[layer - 1]))
            deltas[layer - 1] = (t.T * m).T
        return map(lambda a: a[1:], deltas) # remove the bias neuron delta

    def get_last_layer_delta(self, score, y):
        return -(y - vec_tanh(score)) * 2 * vec_der_tanh(score)


if __name__ == "__main__":
    train_data = np.loadtxt('hw4_nnet_train.dat')
    nn = Network(5000, [2, 3, 1], 0.01, 0.1,10,0.5)
    old = time.time()
    nn.mini_batch_SGD(train_data)
    test = np.loadtxt('hw4_nnet_test.dat')
    e_out = nn.evaluate(test[:,:-1],test[:,-1])
    print "e_out rate : "+str(e_out)
    print time.time() - old





    
