import argparse
import time
import sys

import numpy as np

import chainer
from chainer import report, computational_graph
from chainer import optimizers, serializers

import data
import net
from net import AutoEncoder
from net import Regression

parser = argparse.ArgumentParser(description="MNIST Auto-Encoder Trainer in Chainer")
# file loading settings
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
# computation/learning settings
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
# network structure settings
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--activation', '-a', choices=('relu', 'sigmoid'),
                    default='relu', help="activation function")

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch

n_units = args.unit
activation = args.activation

print('MNIST Auto-Encoder in Chainer')
print()

print('activation func: %s' % activation)
print('# unit: %d' % n_units)

print('# minibatch-size: %d' % batchsize)
print('# epoch: %d' % n_epoch)

# prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255

N = data.num_train
x_train, x_test = np.split(mnist['data'], [N])          # pixels
y_train, y_test = np.split(mnist['data'].copy(), [N])   # same pixels for auto-encoding

# initialize model
model = Regression(AutoEncoder(784, n_units, activation))

# initialize optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(0, n_epoch):
    print ('epoch', epoch+1)

    perm = np.random.permutation(N)

    sum_loss = 0

    start = time.time()
    for i in range(0, N, batchsize):
        x = chainer.Variable(np.asarray(x_train[perm[i:i+batchsize]]))
        t = chainer.Variable(np.asarray(y_train[perm[i:i+batchsize]]))

        optimizer.update(model, x, t)

        if epoch == 0 and i == 0:
            with open('graph.dot', 'w') as o:
                variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
                function_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}
                g = computational_graph.build_computational_graph(
                    (model.loss,),
                    variable_style=variable_style,
                    function_style=function_style)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time

    print('train mean loss={}, throughput={} images/sec'.format(sum_loss / N, throughput))

print('save the model')
serializers.save_npz('{}-{}units_batch{}-epoch{}.model'.format(activation, n_units, batchsize, n_epoch), model)

