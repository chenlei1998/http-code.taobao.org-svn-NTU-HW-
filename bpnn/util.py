__author__ = 'carlxie'

import numpy as np
import matplotlib.pyplot as plt
import struct

# remember our old friend ??
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def uniform_rand(r, shape):
    return 2 * r * np.random.random(shape) - r

## tanh is just a scaled and shifted sigmoid function
def tanh(s):
    return 2 * sigmoid(2 * s) - 1


def der_tanh(s):
    t = sigmoid(2 * s)
    return 4 * t * (1 - t)

def rand_pick(X):
    index = int(np.random.random() * len(X))
    return X[index][:-1], X[index][-1]

# vectorize our util function
vec_tanh = np.vectorize(tanh)
vec_der_tanh = np.vectorize(der_tanh)

def sign(v):
    if v > 0:return 1
    else:return -1

def vec_output(index):
    result = np.ones(10)*-1
    result[index] = 1
    return result

def load_minst(test_imgs_name,test_label_name):
    test_imgs_file = open(test_imgs_name,"rb")
    test_label_file = open(test_label_name,"rb")

    test_set = []
    label_set = []

    try:
        test_imgs = test_imgs_file.read()
        test_labels = test_label_file.read()
    finally:
        test_imgs_file.close()
        test_label_file.close()

    label_index = 0
    magic, numLabels = struct.unpack_from('>II' , test_labels , label_index)
    label_index += struct.calcsize('>II')
    for i in range(numLabels):
        label = struct.unpack_from('>1B' ,test_labels, label_index)
        label_index += struct.calcsize('>1B')
        label_set.append(label)

    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , test_imgs , index)
    index += struct.calcsize('>IIII')
    for i in range(numImages):
        im = struct.unpack_from('>784B' ,test_imgs, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im = im / 255.0
        test_set.append(im)

    test_data = []
    for (im,label) in zip(test_set,label_set):
        nim = np.array(im)
        nlabel = np.array(label)
        test_data.append(np.array(np.append(nim,nlabel)))

    return np.array(test_data)

if __name__ == "__main__":
    #filename = "t10k-images-idx3-ubyte"
    #labelname = "t10k-labels-idx1-ubyte"

    #test = load_minst(filename,labelname)

    #np.save("test.dat",test)

    #filename = "train-images-idx3-ubyte"
    #labelname = "train-labels-idx1-ubyte"

    #train = load_minst(filename,labelname)
    #np.save("train.dat",train)
    pass
