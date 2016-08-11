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

# computation/learning settings
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--noise', '-n', default=0, type=float,
                    help='ratio for adding noise')
# network structure settings
parser.add_argument('--filter', '-f', default=10, type=int,
                    help='number of filters')
parser.add_argument('--filtersize', '-s', default=9, type=int,
                    help='filter size')
parser.add_argument('--unit', '-u', default=30, type=int,
                    help='number of units')
parser.add_argument('--activation', '-a', choices=('relu', 'sigmoid'),
                    default='relu', help="activation function")

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch

n_filters = args.filter
n_units = args.unit
filter_size = args.filtersize
activation = args.activation

print('MNIST Convolutional AutoEncoder in Chainer')
print()

print('[settings]')
print('GPU: %d' % args.gpu)
print('# filter: %d' % n_filters)
print('# unit: %d' % n_units)
print('filter size: %d' % filter_size)
print('activation func: %s' % activation)

print('# minibatch-size: %d' % batchsize)
print('# epoch: %d' % n_epoch)
print('noise ratio: %f' % args.noise)

# GPU setup
xp = np
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    xp = cuda.cupy

# prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255

N = data.num_train
y_train, y_test = np.split(mnist['data'].copy(), [N])   # same pixels for auto-encoding

# add noise
if args.noise > 0:
    for data in mnist['data']:
        perm = np.random.permutation(mnist['data'].shape[1])[:int(mnist['data'].shape[1] * args.noise)]
        data[perm] = 0.0
x_train, x_test = np.split(mnist['data'], [N])          # pixels

# convert to tensor repr
x_train = x_train.reshape((len(x_train), 1, 28, 28))
x_test = x_test.reshape((len(x_test), 1, 28, 28))
y_train = y_train.reshape((len(y_train), 1, 28, 28))
y_test = y_test.reshape((len(y_test), 1, 28, 28))


# initialize model
model = Regression(AutoEncoder(28, n_filters, n_units, filter_size, activation))

# initialize optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(0, n_epoch):
    print ('epoch', epoch+1)

    perm = np.random.permutation(N)
    permed_x = xp.array(x_train[perm])
    permed_y = xp.array(y_train[perm])

    sum_loss = 0

    start = time.time()
    for i in range(0, N, batchsize):
        x = chainer.Variable(permed_x[i:i+batchsize])
        t = chainer.Variable(permed_y[i:i+batchsize])

        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time

    print('train mean loss={}, throughput={} images/sec'.format(sum_loss / N, throughput))

print('save the model')
serializers.save_npz('{}-{}units_batch{}-epoch{}_noise{}.model'.format(activation, n_units, batchsize, n_epoch, args.noise), model)
