import argparse

import sys
import time
from datetime import datetime

import numpy as np

import chainer
from chainer import report
from chainer import optimizers, serializers
from chainer import cuda

import data
import net
from net import AutoEncoder, StackedAutoEncoder, Regression

parser = argparse.ArgumentParser(description="MNIST Stacked Auto-Encoder Trainer in Chainer")
# file loading settings
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
# computation/learning settings
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch_pre', '-p', default=20, type=int,
                    help='number of epochs for pre-training')
parser.add_argument('--epoch_fine', '-f', default=20, type=int,
                    help='number of epochs for fine-tuning')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--noise', '-n', default=0, type=float,
                    help='ratio for adding noise')
# network structure settings
parser.add_argument('--unit', '-u', default='1000,500,250,2',
                    help='number of units (comma-separated)')
parser.add_argument('--activation', '-a', choices=('relu', 'sigmoid'),
                    default='relu', help="activation function")

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch_pre
n_epoch_fine = args.epoch_fine

n_units = list(map(int, args.unit.split(',')))
activation = args.activation

print('MNIST Stacked Auto-Encoder in Chainer')

print()
print('[settings]')
print('- activation func: %s' % activation)
print('- structure: ', n_units)

print('- GPU: %d' % args.gpu)
print('- minibatch-size: %d' % batchsize)
print('- epoch (pre-training): %d' % n_epoch)
print('- epoch (fine-tuning): %d' % n_epoch_fine)
print('- noise ratio: %f' % args.noise)

# GPU setup
xp = np
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    xp = cuda.cupy

# prepare dataset
print()
print('# loading MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(xp.float32)
mnist['data'] /= 255

N = data.num_train
x_train, x_test = xp.split(mnist['data'], [N])
label_train, label_test = xp.split(mnist['target'], [N])

print('- number of training data: {}'.format(N))
print('done.')

# prepare layers
aes = []
for idx in range(len(n_units)):
    n_in = n_units[idx-1] if idx > 0 else 28*28
    n_out = n_units[idx]
    ae = AutoEncoder(n_in, n_out, activation)
    aes.append(ae)

# layer-wise pre-training
print()
print('# layer-wise pre-training')
for idx in range(len(aes)):
    ae = aes[idx]
    if args.gpu >= 0:
        ae.to_gpu()

    print('training layer #{} ({} -> {})'.format(idx+1, ae.n_in, ae.n_out))

    # type of train_data : np.ndarray
    if idx == 0:
        train_data = xp.asarray(x_train)
    else:
        train_data = train_data_for_next_layer

    # ToDo: adding noise to train_data
    input_data = train_data.copy() # train_data + noise

    # prepare regression model and optimizer
    model = Regression(ae)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # training loop
    for epoch in range(0, n_epoch):
        print('  epoch {}'.format(epoch+1))
        perm = xp.random.permutation(N)

        sum_loss = 0
        start = time.time()
        for i in range(0, N, batchsize):
            x = chainer.Variable(input_data[perm[i:i+batchsize]])
            y = chainer.Variable(train_data[perm[i:i+batchsize]])

            optimizer.update(model, x, y)
            sum_loss += float(model.loss.data) * len(y.data)
        end = time.time()
        throughput = N / (end - start)
        print('    train mean loss={}, throughput={} data/sec'.format(sum_loss / N, throughput))

    # prepare train data for next layer
    x = chainer.Variable(train_data)
    train_data_for_next_layer = ae.encode(x, train=False).data
print('done.')

# whole network fine-tuning
aes_copy = []
for ae in aes:
    aes_copy.append(ae.copy())
model = Regression(StackedAutoEncoder(aes_copy))
if args.gpu >= 0:
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)

print()
print('# whole network fine-tuning')
for epoch in range(0, n_epoch_fine):
    print ('  epoch', epoch+1)
    perm = xp.random.permutation(N)

    sum_loss = 0
    start = time.time()
    for i in range(0, N, batchsize):
        x = chainer.Variable(x_train[perm[i:i+batchsize]])
        y = chainer.Variable(x_train[perm[i:i+batchsize]])

        optimizer.update(model, x, y)
        sum_loss += float(model.loss.data) * len(y.data)
    end = time.time()
    throughput = N / (end - start)
    print('    train mean loss={}, throughput={} data/sec'.format(sum_loss / N, throughput))
print('done.')

print()
print('save the model')
serializers.save_npz('sae_{}.model'.format(datetime.now().strftime('%Y%m%d%H%M')), model)

