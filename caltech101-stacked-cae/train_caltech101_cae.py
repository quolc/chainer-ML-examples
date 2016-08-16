import argparse
import time
from datetime import datetime as dt
import sys

import numpy as np

import chainer
from chainer import report, computational_graph
from chainer import optimizers, serializers, cuda

import data_caltech101 as data
import net
from net import DCAE
from net import Regression

parser = argparse.ArgumentParser(description="Caltech101 Auto-Encoder Trainer in Chainer")

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
if fc_units == [0]:
    fc_units = []

activation = args.activation

print('Caltech101 Deep Convolutional Auto Encoder in Chainer')
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
print('load caltech-101 dataset (227x227 cropped)')
caltech = data.loadCaltech101()
x_train = caltech['x_train'].astype('float32')
x_train /= 255
x_test = caltech['x_test'].astype('float32')
x_test /= 255
y_train = x_train.copy()
y_test = x_test.copy()

N = data.num_train
N_test = data.num_test
print('N: {}, N_test: {}'.format(N, N_test))

# add noise
if args.noise > 0:
    for data in x_train:
        for ch in range(0, 3):
            perm = np.random.permutation(x_train.shape[1])[:int(x_train.shape[1] * args.noise)]
            data[ch][perm] = 0.0

# convert to tensor repr
x_train = x_train.reshape((len(x_train), 3, 227, 227))
x_test = x_test.reshape((len(x_test), 3, 227, 227))
y_train = y_train.reshape((len(y_train), 3, 227, 227))
y_test = y_test.reshape((len(y_test), 3, 227, 227))
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
model = Regression(DCAE(227, layers, fc_units, activation))

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
    permed_x = np.array(x_train[perm])
    permed_y = np.array(y_train[perm])

    sum_loss = 0

    start = time.time()
    for i in range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(permed_x[i:i+batchsize]))
        y = chainer.Variable(xp.asarray(permed_y[i:i+batchsize]))
        if len(x.data) < batchsize:
            break

        optimizer.update(model, x, y)
        sum_loss += float(model.loss.data) * y.data.shape[0]
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time

    print('train mean loss={}, throughput={} images/sec'.format(sum_loss / N, throughput))
    last_train_accuracy = sum_loss / N

    sum_loss = 0
    for i in range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i+batchsize]))
        y = chainer.Variable(xp.asarray(y_test[i:i+batchsize]))
        sum_loss += model(x, y, False).data * len(y.data)
    print('test mean loss={}'.format(sum_loss / N_test))
    last_test_accuracy = sum_loss / N_test
    sys.stdout.flush()

print('save the model')
modelname = '{}.model'.format(dt.now().strftime('%m%d%H%M'))
serializers.save_npz(modelname, model)

with open('arch.txt', 'a') as f:
    f.write('[%s]\n' % modelname)
    f.write('conv:\t' + str(layers) + '\n')
    f.write('fc:\t' + str(fc_units) + '\n')
    f.write('alpha:\t' + str(args.alpha) + '\n')
    f.write('epoch:\t' + str(n_epoch) + '\n')
    f.write('noise:\t' + str(args.noise) + '\n\n')
    f.write('train:\t{}, test:\t{}'.format(last_train_accuracy, last_test_accuracy))
    f.write('\n\n')
