import numpy as np

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def tanh(s):
    return 2*sigmoid(2*s)-1

def der_tanh(s):
    t = sigmoid(2*s)
    return 4 * t * (1 - t)

# vectorize our util function
vec_tanh = np.vectorize(tanh)
vec_der_tanh = np.vectorize(der_tanh)

def uniform_rand(r,shape):
    return 2 * r * np.random.random(shape) - r

class Network():
    ###
    ##  use SGD to train our network T times
    ##  for a 2 X 3 X 3 network,shape would be like [2,3,3]
    ##  eta is the learning rate
    ##  r just the range we use the sample the initial weights
    ###
    def __init__(self,T,shape,eta,r):
        self.T = T
        self.eta = eta
        self.shape = np.array(shape)
        self.W = [uniform_rand(r,(shape[i]+1,shape[i+1])) for i in range(len(shape)-1)]
	
    ###
    ##  forward compute the score of the final layer
    ###
    def score(self,x,weights):
	    activation = x
	    for w in weights:
	        activation = np.append(1,activation) # add the bias term
	        activation = vec_tanh(activation.dot(w))
	    return activation


    def cost(self,x,y,weights):
        return (y - self.score(x,weights)[0])**2
    
    ###
    ##  for binary clssfication problem use the sign of the score of the final layer 
    ##  for mult-ti class problem,use argmax(score) to predict instead
    ##  note: we use tanh on the last layer,so its score has been transformed
    ###
    def sign(self,v):
	    if v > 0:return 1
	    else:return -1

    def predict(self,x,weights):
	    return self.sign(self.score(x,weights)[0])

    ###
    ##  test function
    ###
    def test(self,X,Y):
        return sum([self.predict(X[i],self.W) != Y[i]
                    for i in range(len(X))]) / float(len(X))

    ###
    ##  SGD is short for stochastic gradient descent
    ##  for any function f(x;w),x is the input,w is parameter:
    ##  1. compute the gradient delta (df/dw)
    ##  2. update the weight w = w - alpha * delta
    ##     (alpha is the learning rate)
    ###
    def SGD_train(self,X,Y):
        for t in range(self.T):
            x,y = self.rand_pick(X,Y) # that is why it called stochastic
            activations,scores = self.forward_compute(x,y)

            ## backward pass, compute the delta of each neuron
            ## after the deltas are computed,use them to compute
            ## the gradient of each weight       
            deltas = self.backward(scores,x,y)
            grads = self.compute_gradient(activations,deltas)
            ## finally update all the weights
            self.update_weights(grads)

    def cal_grad(self,x,y):
        activations,scores = self.forward_compute(x,y)
        deltas = self.backward(scores,x,y)
        grads = self.compute_gradient(activations,deltas)
        return grads
    
    def forward_compute(self,x,y):
        ## first we forward pass to compute and cache the
        ## score of each layer, in the meanwhile we store
        ## all the activation of each neuron for later use
        scores = []
        activations = []
        activation = x
        for w in self.W:
            activation = np.append(1,activation) # add bias term
            activations.append(activation)
            score = activation.dot(w)  
            scores.append(score)
            activation = vec_tanh(score) # feature transforms
        return activations,scores

    def compute_gradient(self,activations,deltas):
        gradients = [np.empty(w.shape) for w in self.W]
        for i in range(len(self.W)):
            deltas[i].shape = (deltas[i].shape[0],1)
            activations[i].shape = (1,activations[i].shape[0])
            gradients[i] = deltas[i].dot(activations[i]).T
        return gradients

    def update_weights(self,gradients):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.eta * gradients[i]

    def backward(self,scores,x,y):
        deltas = [None for size in self.shape[1:]]
        init_delta = self.get_last_layer_delta(scores[-1],y)
        deltas[-1] = np.append(1,init_delta) # just for convenience
        for layer in range(len(deltas)-1,0,-1):
            t = self.W[layer].dot(deltas[layer][1:])
            m = vec_der_tanh(np.append(1,scores[layer-1]))
            deltas[layer-1] = (t.T*m).T
        return map(lambda a : a[1:],deltas)

    def get_last_layer_delta(self,score,y):
        return -(y - vec_tanh(score)) * 2 * vec_der_tanh(score)
    
    def rand_pick(self,X,Y):
        index = int(np.random.random() * len(X))
        return X[index],Y[index]

if __name__ == "__main__":
    train = np.loadtxt('hw4_nnet_train.dat')
    nn = Network(30000,[2,3,1],0.01,0.1)
    nn.SGD_train(train[:,:-1],train[:,-1])
    rate = nn.test(train[:,:-1],train[:,-1])

    print "error_in %f " %rate
    
#    test = np.loadtxt('hw4_nnet_test.dat')

#    e_out = []
#    for enta in [0.001,0.01,0.1,1,10]:
#        err = 0.0
#        for t in range(80):
#            print "start >>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(t)
#            nn = Network(30000,[2,3,1],enta,0.1)
#            nn.SGD_train(train[:,:-1],train[:,-1])
#            err = err + nn.test(test[:,:-1],test[:,-1])
#        e_out.append(err/80)
#    print "Q13"
#    print e_out



    
