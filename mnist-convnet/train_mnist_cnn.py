import argparse
import sys, time, datetime

import numpy as np

import chainer
from chainer import optimizers, serializers, cuda

import data
import net

parser = argparse.ArgumentParser(description="MNIST Convolutional Neural Network Training")
# file loading

# computation / learning settings
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs for training')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--optimizer', '-o', choices=('adam', 'momentumsgd'),
                    default='adam', help='optimizer (adam or momentumsgd)')
parser.add_argument('--learningrate', type=float, default=0.01,
                    help='learning rate (momentum SGD only)')
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha value (Adam only)')

# structure settings
# ToDo: prepare setting file

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch

print()
print('[settings]')
print('- GPU: %d' % args.gpu)
print('- minibatch-size: %d' % batchsize)
print('- epoch: %d' % n_epoch)

print('- optimizer: %s' % args.optimizer)
if args.optimizer == 'momentumsgd':
    print('- learning rate: %f' % args.learningrate)
if args.optimizer == 'adam':
    print('- alpha: %f' % args.alpha)

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
mnist['target'] = mnist['target'].astype(xp.int32)

N = data.num_train
N_test = data.num_test
x_train, x_test = xp.split(mnist['data'], [N])
y_train, y_test = xp.split(mnist['target'], [N])

# convert to tensor representation
x_train = x_train.reshape((len(x_train), 1, 28, 28))
x_test = x_test.reshape((len(x_test), 1, 28, 28))

print('- number of training data: %d' % N)
print('- number of test data: %d' % N_test)
print('done.')

# prepare network
model = net.Classifier(net.ConvNet(28, 28, 1, 8, 256, 10))

# initialize optimizer
if args.optimizer == 'adam':
    optimizer = optimizers.Adam(args.alpha)
if args.optimizer == 'momentumsgd':
    optimizer = optimizers.MomentumSGD(args.learningrate)
optimizer.setup(model)

# training loop
print()
print('start learning')
if args.gpu >= 0:
    model.to_gpu()
for epoch in range(0, n_epoch):
    print('epoch', epoch+1)

    perm = xp.random.permutation(N)
    permed_data = xp.array(x_train[perm])
    permed_target = xp.array(y_train[perm])

    sum_accuracy = 0
    sum_loss = 0

    start = time.time()
    for i in range(0, N, batchsize):
        x = chainer.Variable(permed_data[i:i+batchsize])
        y = chainer.Variable(permed_target[i:i+batchsize])

        optimizer.update(model, x, y)
        sum_loss += float(model.loss.data) * len(y.data)
        sum_accuracy += float(model.accuracy.data) * len(y.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time

    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

print('save the model') 
serializers.save_npz('linear.model', model)

