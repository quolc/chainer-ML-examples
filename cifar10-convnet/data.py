import pickle
import numpy as np
import sys

path_prefix = 'cifar-10-batches-py'
files_train = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
files_test = ['test_batch']

num_train = 50000
num_test = 10000

def loadCifar10():
    x_train, x_test = None, None
    y_train, y_test = None, None

    for file in files_train:
        path = path_prefix + '/' + file
        print('opening ' + path)
        with open(path, 'rb') as f:
            if sys.version_info.major == 2: # python 2
                d = pickle.load(f)
            else: # python 3
                d = pickle.load(f, encoding='latin-1')

            if x_train is None:
                x_train = d['data']
                y_train = d['labels']
            else:
                x_train = np.vstack((x_train, d['data']))
                y_train = y_train + d['labels']

    for file in files_test:
        path = path_prefix + '/' + file
        print('opening ' + path)
        with open(path, 'rb') as f:
            if sys.version_info.major == 2:
                d = pickle.load(f)
            else:
                d = pickle.load(f, encoding='latin-1')
            if x_test is None:
                x_test = d['data']
                y_test = d['labels']
            else:
                x_test = np.vstack((x_test, d['data']))
                y_test = y_test + d['labels']

    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
