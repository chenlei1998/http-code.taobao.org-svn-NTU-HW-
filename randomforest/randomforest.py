from decisiontree import *
import numpy as np

def sample(x,y):
    x_new = []
    y_new = []
    size = x.shape[0]
    for i in range(size):
        index = int(np.random.random()*size)
        x_new.append(x[index])
        y_new.append(y[index])
    return np.array(x_new),np.array(y_new)

def train_rf(x,y,T):
    trees = []
    for t in range(T):
        x_new,y_new = sample(x,y)
        trees.append(train_decision_tree(x_new,y_new))
    return trees

def predict_rf(x,model):
    result = []
    for tree in model:
        if decision_stump(tree[0],x[tree[1]],tree[2]) == 1:
            result.append(tree[3])
        else:
            result.append(tree[4])
    return majority(np.array(result))

def test_rf(x,y,model):
    return sum([1 for i in range(len(y)) if predict_rf(x[i],model) != y[i]]) / float(len(y))

def train_rf_weak_learner(x,y,T):
    trees = []
    for t in range(T):
        x_new,y_new = sample(x,y)
        tree = argmin_decision_stump_clever(x_new,y_new)
        trees.append(tree)
    return trees

if __name__ == "__main__":
    (x,y) = load_file('hw3_train.dat')
    (x_t,y_t) = load_file('hw3_test.dat')

    E_g_in = []
    E_G_in = []
    E_G_out = []
    
    for c in range(100):
        print "train round "+str(c)+" ....."
        forest = train_rf_weak_learner(x,y,300)
        #for tree in forest:
        #    E_g_in.append(test_decision_tree(x,y,tree))
        E_G_in.append(test_rf(x,y,forest))
        E_G_out.append(test_rf(x_t,y_t,forest))
        print "train round "+str(c)+" ..... end"
    
#    print "E_g_in : "+str(np.mean(E_g_in))
    print "E_G_in : "+str(np.mean(E_G_in))
    print "E_G_out : "+str(np.mean(E_G_out))
     
    
    
