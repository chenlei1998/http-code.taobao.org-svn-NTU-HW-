import numpy as np

class kNN():

    def __init__(self,k):
        self.k = k
    
    def train(self,x,y):
        self.x = x
        self.y = y

    def predict(self,x):
        distances = map(sum,(self.x - x)**2)
        pairs = zip(distances,self.y)
        pairs.sort()
        return self.majority([pairs[i][1] for i in range(self.k)])

    def majority(self,y):
        y = np.array(y)
        if len(y[y==-1]) > len(y)/2:
            return -1
        else:
            return 1

    def test(self,x,y):
        return sum([self.predict(x[i]) != y[i] for i in range(len(y))]) / float(len(y))

if __name__ == "__main__":
    knn = kNN(1)
    data = np.loadtxt('hw4_knn_train.dat')
    x = data[:,0:-1]
    y = data[:,-1]
    knn.train(x,y)
    print "1NN Ein "+str(knn.test(x,y))

    test = np.loadtxt('hw4_knn_test.dat')
    print "1NN Eout "+str(knn.test(test[:,0:-1],test[:,-1]))

    knn = kNN(5)
    data = np.loadtxt('hw4_knn_train.dat')
    x = data[:,0:-1]
    y = data[:,-1]
    knn.train(x,y)
    print "5NN Ein "+str(knn.test(x,y))

    test = np.loadtxt('hw4_knn_test.dat')
    print "5NN Eout "+str(knn.test(test[:,0:-1],test[:,-1]))
    
    
    
        
