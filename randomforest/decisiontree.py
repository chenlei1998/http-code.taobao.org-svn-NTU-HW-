
import numpy as np

def load_file(file_name):
    data = np.loadtxt(file_name)
    data_x = np.array([data[:,0],data[:,1]]).T
    data_y = data[:,2]
    return (data_x,data_y)

def get_branch_part(s,i,threshold,x,y):
    part1,part2 = ([],[]),([],[])
    for index in range(len(x)):
        if decision_stump(s,x[index][i],threshold) == 1:
            part1[0].append(x[index])
            part1[1].append(y[index])
        else:
            part2[0].append(x[index])
            part2[1].append(y[index])
    return (part1,part2)

def cal_purity(s,i,threshold,x,y):
    (p1,p2) = get_branch_part(s,i,threshold,x,y)
    return sum([len(p1[1])*gini_index(p1[1]),len(p2[1])*gini_index(p2[1])])

#naively implementation takes O(N^2)
def argmin_decision_stump(train_x,train_y):
    best_s,best_feature,best_theta = 0,0,0
    best_purity = 10000000000 
    for i in range(train_x.shape[1]):
        temp = train_x[:,i].copy()
        temp.sort()
        thresholds = [temp[0]-1] + [(temp[k]+temp[k+1])/float(2) for k in range(len(temp)-1)]
        for threshold in thresholds:
            purity = cal_purity(1,i,threshold,train_x,train_y)
            if best_purity > purity:
                best_purity = purity
                best_s = 1
                best_threshold = threshold
                best_i = i

    return (best_s,best_i,best_threshold,best_purity) 

def gini_index(y):
    y = np.array(y)
    if len(y) == 0:
        return 0
    else:
        t = len(y[y==1])/float(len(y))
        return 2*(t-t**2)

#smart implementation takes O(Nlog(N))
def argmin_decision_stump_clever(train_x,train_y):
    best_purity = 10000000000
    n_train = len(train_y)
    best_threshold ,best_i = 0,0
    for i in range(train_x.shape[1]):
        temp_x = train_x[:,i].copy()
        temp_y = train_y.copy()
        temp = zip(temp_x,temp_y)
        temp.sort()
        
        p1_num = [len(train_y[train_y > 0])]
        p2_num = [0]
        threshold_base = temp[0][0] - 1000 # less than any feature value
        thresholds = [(temp[k][0]+temp[k+1][0])/2 for k in range(len(temp)-1)]
        for j in range(len(thresholds)):
            if temp[j][1] == 1: 
                p1_num.append(p1_num[j]-1)
                p2_num.append(p2_num[j]+1)
            else:
                p1_num.append(p1_num[j])
                p2_num.append(p2_num[j])

        purities = [my_gini_index(p1_num[0]/float(n_train))*n_train]
        for size in range(n_train-1,0,-1):
            purities.append(my_gini_index(p1_num[n_train-size]/float(size))*size
                            + my_gini_index(p2_num[n_train-size]/float((n_train-size)))*(n_train-size))
        
        cur_best = min(purities)
        if best_purity > cur_best:
            best_purity = cur_best
            best_i = i
            thresholds = [threshold_base] + thresholds
            best_threshold = thresholds[purities.index(best_purity)]

    (p1,p2) = get_branch_part(1,best_i,best_threshold,train_x,train_y)    

    return (1,best_i,best_threshold,majority(np.array(p1[1])),majority(np.array(p2[1]))) 

def my_gini_index(mu):
    return 2*(mu-mu**2)

def decision_stump(s,x,threshold):
    def sign(x,threshold):
        if x >= threshold:return 1
        else:return -1
    return s * sign(x,threshold)   

def majority(y):
    if len(y[y==-1]) > len(y)/2:
        return -1
    else:
        return 1

def meet_termination_criteria(x,y):
    #print "x len: "+str(len(x)) + " gini_index: "+str(gini_index(y))
    return gini_index(y) == 0 or all([all(x[0]==ele) for ele in x])

def train_decision_tree(x,y):
    if len(x) == 0 or len(y) == 0:
        return None
    x,y = np.array(x),np.array(y)
    if meet_termination_criteria(x,y):
        return majority(y)
    else:
        head = argmin_decision_stump_clever(x,y)
        (p1,p2) = get_branch_part(head[0],head[1],head[2],x,y)
        left,right = None,None
        if len(p1[0]) >= 1:
            left = train_decision_tree(p1[0],p1[1])
        if len(p2[0]) >= 1:
            right = train_decision_tree(p2[0],p2[1])
        return (head,left,right)

def predict(x,model):
    if type(model) == tuple:
        (head,left,right) = model
        if decision_stump(head[0],x[head[1]],head[2]) == 1:
            return predict(x,left)
        else:
            return predict(x,right)
    else:
        return model

def test_decision_tree(x,y,model):
    #print [predict(x[i],model)for i in range(len(y))]
    return sum([1 for i in range(len(y)) if predict(x[i],model) != y[i]]) / float(len(y))

if __name__ == "__main__":
    (x,y) = load_file('hw3_train.dat')
    model = train_decision_tree(x,y)
    error_in = test_decision_tree(x,y,model)
    print "error_in : "+str(error_in)
    (x_t,y_t) = load_file('hw3_test.dat')
    error_out = test_decision_tree(x_t,y_t,model)
    print "error_out : "+str(error_out)
