__author__ = 'carlxie'

from util import *
import time

class BPNetwork():
    ###
    ##  for a 2 X 3 X 3 network,shape would be like [2,3,3]
    ##  r just the range we use the sample the initial weights
    ###
    def __init__(self,shape,r):
        self.weights = [uniform_rand(r, (shape[i], shape[i + 1])) for i in range(len(shape) - 1)]
        self.biases = [np.zeros((1,shape[i + 1])) for i in range(len(shape) - 1)]

    def forward_prop(self,x,weights,biases):
        a = x
        for w,b in zip(weights,biases):
            a = vec_tanh(a.dot(w) + b)
        return a

    def predict(self,x,weights,biases):
        return np.argmax(self.forward_prop(x,weights,biases))

    def cost(self, x, y, weights,biases,mini_batch_size,lamb):
        return sum(vec_output(y) - self.forward_prop(x, weights,biases)) ** 2 + self.reg_cost(mini_batch_size,lamb)

    def reg_cost(self,mini_batch_size,lamb):
        W = np.array(self.weights)**2
        loss = 0.0
        for w in W:
            loss = loss + sum(sum(w))
        return lamb * loss / mini_batch_size

    def evaluate(self, X, Y):
        return sum([self.predict(X[i], self.weights,self.biases) != Y[i]
                    for i in range(len(X))]) / float(len(X))

    ###
    ##  SGD is short for stochastic gradient descent
    ##  for any function f(x;w),x is the input,w is parameter:
    ##  1. compute the gradient delta (df/dw)
    ##  2. update the weight w = w - eta * delta
    ##     (eta is the learning rate)
    ##  args:
    ##     T is training times
    ##     lamb is regularization parameter lambda
    ##     eta is the learning rate
    ###
    def SGD_train(self,training_data,T,eta,lamb):
        n = len(training_data)
        for t in range(T):
            x, y = rand_pick(training_data)  # that is why it called stochastic
            ## all the dirty work get done here
            weight_grads,bias_grads = self.calculate_analytic_grads(x,y)
            ## finally update all the weights
            self.update_weights(weight_grads,bias_grads,eta,lamb,1,n)

    ###
    ##  no big change from SGD, just we partition X into small
    ##  patches which size is determined by mini_batch_size
    ##  and then use the average of those patches' gradients
    ##  to update our weights
    ###
    def mini_batch_SGD(self,training_data,T,eta,lamb,mini_batch_size):
        n = len(training_data)
        for t in range(T):
            print "epoch t "+str(t)
            # first we random shuffle our data
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                    for k in xrange(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lamb,n)

    def update_mini_batch(self,mini_batch,eta,lamb,n):
        sum_w_grads = [np.zeros(w.shape) for w in self.weights]
        sum_b_grads = [np.zeros(b.shape) for b in self.biases]
        for data in mini_batch:
            w_grads,b_grads = self.calculate_analytic_grads(data[:-1],data[-1])
            sum_w_grads = self.sum_grads(sum_w_grads,w_grads)
            sum_b_grads = self.sum_grads(sum_b_grads,b_grads)

        self.update_weights(sum_w_grads,sum_b_grads,eta,lamb,len(mini_batch),n)

    def sum_grads(self,g1,g2):
        for j in range(len(self.weights)):
            g1[j] = g1[j] + g2[j]
        return g1

    def update_weights(self,w_grads,b_grads,eta,lamb,mini_batch_size,n):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]*(1 - eta * lamb / n) - eta * w_grads[i] / mini_batch_size
            self.biases[i] = self.biases[i] - eta * b_grads[i] / mini_batch_size

    ###
    ##  use multiprocess to accelerate gradient compute
    ##
    ###

    def calculate_analytic_grads(self, x, y):
        x.shape = (1,x.shape[0])
        activations, scores = self.compute_activations_scores(x)
        ## backward pass, compute the delta of each neuron
        ## after the deltas are computed,use them to compute
        ## the gradient of each weight
        deltas = self.back_propagation(scores, y)
        ## grad = delta * a , remember?
        w_grads,b_grads = self.compute_gradient(activations, deltas)
        return w_grads,b_grads

    def compute_activations_scores(self, x):
        ## first we forward pass to compute and cache the
        ## score of each layer, in the meanwhile we store
        ## all the activation of each neuron for later use
        scores = []
        activations = []
        activation = x
        for w,b in zip(self.weights,self.biases):
            activations.append(activation)
            score = activation.dot(w) + b
            scores.append(score)
            activation = vec_tanh(score)  # feature transforms
        return activations, scores

    def compute_gradient(self, activations, deltas):
        w_grads = [np.empty(w.shape) for w in self.weights]
        b_grads = [np.empty(b.shape) for b in self.biases]
        for i in range(len(self.weights)):
            w_grads[i] = deltas[i].dot(activations[i]).T
            b_grads[i] = deltas[i].T
        return w_grads,b_grads

    def back_propagation(self, scores, y):
        deltas = [None for i in range(len(self.weights))]
        deltas[-1] = self.get_last_layer_delta(scores[-1], y).T
        ## backward compute the deltas,note the input layer is not included
        ## deltas[layer-1] = weights[layer-1].dot(deltas[layer]) * der_tanh(score[layer-1])
        for layer in range(len(deltas) - 1, 0, -1):
            deltas[layer - 1] = self.weights[layer].dot(deltas[layer]) * vec_der_tanh(scores[layer - 1]).T
        return deltas

    def get_last_layer_delta(self, score, y):
        return -(vec_output(y) - vec_tanh(score)) * 2 * vec_der_tanh(score)

if __name__ == "__main__":
    train_data = np.load('train.dat.npy')
    nn = BPNetwork([784,30,10],0.5)
    old = time.time()
    nn.mini_batch_SGD(train_data,30,0.01,0.5,20)
    test = np.load('test.dat.npy')
    e_out = nn.evaluate(test[:,:-1],test[:,-1])
    print "e_out rate : "+str(e_out)
    print time.time() - old