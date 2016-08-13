import argparse
import time
from datetime import datetime as dt
import sys

import numpy as np

import chainer
from chainer import report, computational_graph
from chainer import optimizers, serializers, cuda

import data
import net
from net import DCAE
from net import Regression

parser = argparse.ArgumentParser(description="MNIST Auto-Encoder Trainer in Chainer")

# computation/learning settings
parser.add_argument('--dry', action="store_true")
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--noise', '-n', default=0, type=float,
                    help='ratio for adding noise')
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha (only for Adam)')
# network structure settings
parser.add_argument('--channels', '-c', default='20,20',
                    help='number of filters')
parser.add_argument('--filtersizes', '-s', default='5,5',
                    help='filter size')
parser.add_argument('--pads', '-d', default='0',
                    help='pad sizes')
parser.add_argument('--poolings', '-p', default='0',
                    help='pooling sizes')
parser.add_argument('--fcunits', '-u', default='100,10',
                    help='number of units')
parser.add_argument('--activation', '-a', choices=('relu', 'sigmoid'),
                    default='relu', help="activation function")

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch

channels = list(map(int, args.channels.split(',')))
filter_sizes = list(map(int, args.filtersizes.split(',')))
pads = list(map(int, args.pads.split(',')))
poolings = list(map(int, args.poolings.split(',')))
fc_units = list(map(int, args.fcunits.split(',')))

activation = args.activation

print('MNIST Deep Convolutional Auto Encoder in Chainer')
print()

print('[settings]')
print('channels', channels)
print('filter_sizes', filter_sizes)
print('pads', pads)
print('poolings', poolings)
print('fc_units', fc_units)

print('activation func: %s' % activation)

print('minibatch-size: %d' % batchsize)
print('epoch: %d' % n_epoch)
print('noise ratio: %f' % args.noise)
print('GPU: %d' % args.gpu)
print()

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
N_test = data.num_test
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
print('done.')
print()

# initialize model
layers = []
for i in range(len(channels)):
    if len(pads) <= i:
        pads.append(0)
    if len(poolings) <= i:
        poolings.append(0)
    layer = (filter_sizes[i], channels[i], pads[i], poolings[i])
    layers.append(layer)
model = Regression(DCAE(28, layers, fc_units, activation))

# initialize optimizer
optimizer = optimizers.Adam(args.alpha)
optimizer.setup(model)
if args.dry:
    sys.exit()
if args.gpu >= 0:
    model.to_gpu()

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

    sum_loss = 0
    test_x = xp.array(x_test)
    test_y = xp.array(y_test)

    for i in range(0, N_test, batchsize):
        x = chainer.Variable(test_x[i:i+batchsize])
        y = chainer.Variable(test_y[i:i+batchsize])
        sum_loss += model(x, y, False).data * len(y.data)
    print('test mean loss={}'.format(sum_loss / N_test))
    sys.stdout.flush()

print('save the model')
modelname = '{}.model'.format(dt.now().strftime('%m%d%H%M'))
serializers.save_npz(modelname, model)

with open('arch.txt', 'a') as f:
    f.write(str(modelname) + '\n')
    f.write('conv:\t' + str(layers) + '\n')
    f.write('fc:\t' + str(fc_units) + '\n')
    f.write('alpha:\t' + str(args.alpha) + '\n')
    f.write('epoch:\t' + str(n_epoch) + '\n')
    f.write('noise:\t' + str(args.noise) + '\n\n')
