from math import *
import numpy as np

def load_file(file_name):
    data = np.loadtxt(file_name)
    data_x = np.array([data[:,0],data[:,1]]).T
    data_y = data[:,2]
    return (data_x,data_y)

def desion_stump(s,x,threshold):
    def sign(x,threshold):
        if x >= threshold:return 1
        else:return -1
    return s * sign(x,threshold)   

def cal_err(i,s,threshold,data_weights,train_y):
    return sum([data_weights[j] for j in range(len(train_y))
                if desion_stump(s,train_x[j,i],threshold) != train_y[j]]) / sum(data_weights)

### naively implementation takes O(N^2)
def get_decision_stump(train_x,train_y,data_weights):
    best_err,best_s,best_threshold,best_i = len(train_x[:,0]),0,0,0
    for i in range(train_x.shape[1]):
        temp = train_x[:,i].copy()
        temp.sort()
        thresholds = [temp[0]-1] + [(temp[k]+temp[k+1])/2 for k in range(len(temp)-1)]       
        for s in [-1,1]:
            for threshold in thresholds:
                err = cal_err(i,s,threshold,data_weights,train_y)
                if best_err >= err:
                    best_err = err
                    best_s = s
                    best_threshold = threshold
                    best_i = i

    return (best_s,best_i,best_threshold,best_err)

### clever way of implementation takes only O(Nlog(N))!!
def get_decision_stump_clever(train_x,train_y,data_weights):
    
    best_err,best_s,best_threshold,best_i = len(train_x[:,0]),0,0,0

    for i in range(train_x.shape[1]):
        # create a sorted temp list from ith feature of train_x
        # zipped with train_y and data_weights to form a triple list,
        # note that this step is extremely important!
        temp_w = data_weights.copy()
        temp_x = train_x[:,i].copy()
        temp_y = train_y.copy()
        temp = zip(temp_x,temp_y,temp_w)
        temp.sort()

        err_base = sum(data_weights[train_y < 0])

        error_num = len(data_weights[train_y < 0])
        threshold_base = temp[0][0] - 1000 # less than any feature value

        ####
        #       x0    x1     x2      x3  ....    xn
        # t_base   t0     t1     t2      ....tn-1
        ####
        thresholds = [(temp[k][0]+temp[k+1][0])/2 for k in range(len(temp)-1)]
        
        errors = [err_base]

        rec_num = 0
        
        for j in range(len(thresholds)):
            if temp[j][1] == 1: # we made a mistake this time, so punished by its weight accordingly
                errors.append(errors[j] + temp[j][2])
            else:
                errors.append(errors[j] - temp[j][2])
                rec_num = rec_num + 1
        
        assert error_num >= rec_num
        total_weights = sum(data_weights)
        norm_errors = map(lambda x : x / total_weights,errors)
        
        positive_min_err = min(norm_errors)
        negative_min_err = 1-max(norm_errors)

        if positive_min_err < negative_min_err:
            cur_best_err = (positive_min_err,1)
        else:
            cur_best_err = (negative_min_err,-1)
        
        if best_err > cur_best_err[0]:
            (best_err,best_s) = cur_best_err
            best_i = i
            
            temp_err = best_err
            if best_s == -1:
                temp_err = 1 - best_err
            thresholds = [threshold_base] + thresholds
            best_threshold = thresholds[norm_errors.index(temp_err)]
           
    return (best_s,best_i,best_threshold,best_err)
                

def train_adaboost(train_x,train_y,T):
    data_weights = np.ones(len(train_y)) * 0.01 
    sum_weights = [1]
    alphas = []
    models = []
    errs = []
    for t in range(T):
        (s,i,theta,err) = get_decision_stump_clever(train_x,train_y,data_weights)
        errs.append(err)
        err_ratio = sqrt((1-err)/err)
        for index in range(len(data_weights)):
            if(desion_stump(s,train_x[index,i],theta) != train_y[index]):
                data_weights[index] = data_weights[index] * err_ratio
            else:
                data_weights[index] = data_weights[index] / err_ratio
                
        sum_weights.append(sum(data_weights))
        alphas.append(log(err_ratio))
        models.append((s,i,theta))
    return (models,alphas)

def test_adaboost(test_x,test_y,models,alphas):
    def sign(value):
        if value > 0:return 1
        else: return -1
    err = 0
    for index in range(len(test_y)):
        value = sum([alpha * desion_stump(s,test_x[index,i],theta)
                     for (alpha,(s,i,theta)) in zip(alphas,models)])
        if sign(value) != test_y[index]:
            err = err + 1

    return float(err)/len(test_y)

def test_decision_stump(test_x,test_y,model):
    (s,i,theta) = model
    err = sum([1 for j in range(len(test_y)) if desion_stump(s,test_x[j,i],theta) != test_y[j]])
    return float(err)/len(test_y)

if __name__ == '__main__':
    (train_x,train_y) = load_file('hw2_adaboost_train.dat')
    (models,alphas) = train_adaboost(train_x,train_y,300)
    (test_x,test_y) = load_file('hw2_adaboost_test.dat')
    print(test_adaboost(test_x,test_y,models,alphas))
    print(test_decision_stump(test_x,test_y,models[0]))
