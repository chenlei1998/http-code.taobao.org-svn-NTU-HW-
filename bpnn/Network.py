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


def gradient_check():
    train = np.loadtxt('hw4_nnet_train.dat')
    nn = Network(10000,[2,2,1],0.1,0.1)
    grad = nn.SGD_train(train[:,:-1],train[:,-1])
    #num_grad = compute_num_grad(train[:,:-1],train[:,-1])

class Network():
    ###
    ##  use SGD to train our network T times
    ##  for a 2 X 3 X 3 network,shape would be like [2,3,3]
    ##  enta is the learning rate
    ##  r just the range we use the sample the initial weights
    ###
    def __init__(self,T,shape,enta,r):
        self.T = T
        self.enta = enta
        self.shape = np.array(shape)
        self.W = [uniform_rand(r,(shape[i]+1,shape[i+1])) for i in range(len(shape)-1)]


    def predict(self,x):
        activation = x
        for w in self.W:
            activation = np.append(1,activation) # add the bias term
            activation = vec_tanh(activation.dot(w))
        return self.sign(activation[0])

    def sign(self,x):
        if x > 0:return 1
        else:return -1

    def test(self,X,Y):
        return sum([self.predict(X[i]) != Y[i]
                    for i in range(len(X))]) / float(len(X))
            
    
    def SGD_train(self,X,Y):
        gradients = [np.array(w.shape) for w in self.W]
        for t in range(self.T):
            x,y = self.rand_pick(X,Y)

            scores = []
            activations = []
            activation = x
            for w in self.W:
                activation = np.append(1,activation) # add bias term
                activations.append(activation)
                score = activation.dot(w)  
                scores.append(score)
                activation = vec_tanh(score) # feature transforms
            
            deltas = self.backward(scores,x,y)
            self.update_weights(activations,deltas)
            #self.compute_gradient(activations,deltas)
        #return map(lambda a:a/self.T,gradients)

    def compute_gradient(self,activations,deltas):
        for i in range(len(self.W)):
            deltas[i].shape = (deltas[i].shape[0],1)
            activations[i].shape = (1,activations[i].shape[0])
            gradients[i] = gradients[i] + deltas[i].dot(activations[i]).T

           
    def update_weights(self,activations,deltas):
        for i in range(len(self.W)):
            deltas[i].shape = (deltas[i].shape[0],1)
            activations[i].shape = (1,activations[i].shape[0])
            self.W[i] = self.W[i] - self.enta * deltas[i].dot(activations[i]).T

    def backward(self,scores,x,y):
        deltas = [None for size in self.shape[1:]]
        
        init_delta = self.get_last_layer_delta(scores[-1],y)
        deltas[-1] = np.append(1,init_delta) # just for convenience
        
        for layer in range(len(deltas)-1,0,-1):
          #  print "cur w shape"+ str(self.W[layer].shape)
          #  print "cur delta shape " + str(deltas[layer][1:].shape)
            
            t = self.W[layer].dot(deltas[layer][1:])
            m = vec_der_tanh(np.append(1,scores[layer-1]))

          # print "**************************************"
          #  print t.shape
          #  print m.shape

            deltas[layer-1] = (t.T*m).T

        return map(lambda a : a[1:],deltas)


    def get_last_layer_delta(self,score,y):
        return -(y - vec_tanh(score)) * 2 * vec_der_tanh(score)
    
    def rand_pick(self,X,Y):
        index = int(np.random.random() * len(X))
        return X[index],Y[index]
    
if __name__ == "__main__":
    train = np.loadtxt('hw4_nnet_train.dat')
    test = np.loadtxt('hw4_nnet_test.dat')

    e_out = []
    for enta in [0.001,0.01,0.1,1,10]:
        err = 0.0
        for t in range(80):
            print "start >>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(t)
            nn = Network(30000,[2,3,1],enta,0.1)
            nn.SGD_train(train[:,:-1],train[:,-1])
            err = err + nn.test(test[:,:-1],test[:,-1])
        e_out.append(err/80)
    print "Q13"
    print e_out
