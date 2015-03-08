import numpy as np
from numpy import linalg as la
from numpy import matlib as mt

def load(filename):
    data = np.loadtxt(filename)
    train = data[:400,:]
    test = data[400:,:]
    return (train,test)

def parse_data(data):
    x = data[:,:-1]
    y = data[:,-1]
    return x,y

def kernal(x,gamma):
    n = x.shape[0]
    k = mt.empty((n,n))
    for i in range(n):
        for j in range(n):
            k[i,j] = np.exp(-gamma*la.norm(x[i,:]-x[j,:])**2)
    return k

def lssvm_train(x,y,gamma,lamb):
    n = len(y)
    K = kernal(x,gamma)
    lamb_eye = lamb * mt.eye(n,n)
    return (lamb_eye + K).I.dot(y)

def predictor(model,x):
    (betas,gamma,features) = model
    x_norm = np.array(map(la.norm,(features - x)))
    nf = np.exp(-gamma*x_norm**2)
    result = betas.dot(nf)
    if result > 0: return 1
    else: return -1
    
def lssvm_test(x,y,model):
    predict = np.array([predictor(model,feat) for feat in x])
    label = predict > 0
    answer = y > 0
    return sum([1 for i in range(len(y)) if label[i] != answer[i]]) / float(len(y))

if __name__ == '__main__':
    (train,test) = load('hw2_lssvm_all.dat')
    train_x,train_y = parse_data(train)
    test_x,test_y = parse_data(test)
    gamma = [32,2,0.125]
    lamb = [0.001,1,1000]

    e_in = []
    e_out = []
    
    for g in gamma:
        for l in lamb:
            betas = lssvm_train(train_x,train_y,g,l)
            e_in.append(lssvm_test(train_x,train_y,(betas,g,train_x)))
            e_out.append(lssvm_test(test_x,test_y,(betas,g,train_x)))
            print "finish...........gamma: "+str(g)+"  lambda: "+str(l)
    print e_in
    print e_out
