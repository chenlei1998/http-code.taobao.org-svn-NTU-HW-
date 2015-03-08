import numpy as np

class kMeans():
    def __init__(self,k):
        self.k = k

    def train(self,x):
        mu = self.random_init_ceter(x)
        while True:
            clausters = [[] for i in range(self.k)]
            for e in x:
                clausters[np.argmin(map(sum,(mu-e)**2))].append(e)
            newmu = np.array([np.mean(clausters[i],axis=0)
                              for i in range(self.k)])
            if self.norm(newmu,mu) == 0:
                return mu,clausters
            mu = newmu
            
    def random_init_ceter(self,x):
        c = set()
        length = len(x)
        while len(c) != self.k:
            c.add(int(np.random.random() * length))
        return np.array([x[i] for i in c])
    
    def norm(self,u1,u2):
        result = sum(sum((u1-u2)**2))
        return result


    def test(self,clausters,mu,N):
        dis = 0
        for i in range(len(mu)):
            dis = dis + sum(map(sum,(clausters[i]-mu[i])**2))

        return dis/float(N)

if __name__ == "__main__":
    x = np.loadtxt('hw4_kmeans_train.dat')
    count = 500
    c = kMeans(10)
    error = 0
    for t in range(count):
        mu,clausters = c.train(x)
        err = c.test(clausters,mu,len(x))
        error = error + err
        print err
    print "final error "+str(float(error)/count)
        
