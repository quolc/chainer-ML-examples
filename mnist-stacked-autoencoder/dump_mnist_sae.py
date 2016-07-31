import argparse

import sys
import time
import math
from datetime import datetime

import numpy as np

import chainer
from chainer import report
from chainer import optimizers, serializers

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data
import net
from net import AutoEncoder, StackedAutoEncoder, Regression

plt.gray()
def draw_digit_ae(data, n, row, col):
    size = 28
    G = gridspec.GridSpec(row, col)
    sp = plt.subplot(G[n//10, n%10])

    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    sp.pcolor(Z)

    sp.tick_params(labelbottom="off")
    sp.tick_params(labelleft="off")
    sp.axis('scaled')
    sp.axis([0, 28, 0, 28])

parser = argparse.ArgumentParser(description="MNIST Stacked Auto-Encoder Trainer in Chainer")

parser.add_argument('--model', '-m', default='', help='Model of encoder')
parser.add_argument('--num', '-n', default=50, type=int,
                    help='Number of images to dump')
# network structure settings
parser.add_argument('--unit', '-u', default='1000,500,250,2',
                    help='number of units')
parser.add_argument('--activation', '-a', choices=('relu', 'sigmoid'),
                    default='relu', help="activation function")

args = parser.parse_args()

n_units = list(map(int, args.unit.split(',')))
activation = args.activation

print('MNIST Stacked Auto-Encoder in Chainer')

print()
print('[settings]')
print('- activation func: %s' % activation)
print('- structure: ', n_units)

# prepare dataset
print()
print('# loading MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255

N = data.num_train
x_train, x_test = np.split(mnist['data'], [N])
label_train, label_test = np.split(mnist['target'], [N])

print('done.')

# initialize model
aes = []
for idx in range(len(n_units)):
    n_in = n_units[idx-1] if idx > 0 else 28*28
    n_out = n_units[idx]
    aes.append(AutoEncoder(n_in, n_out, activation))

model = Regression(StackedAutoEncoder(aes))
serializers.load_npz(args.model, model)

# process
perm = np.random.permutation(N)
x = chainer.Variable(np.asarray(x_train[perm[0:args.num]]))

# dump
print()
print('# dumping images')
for l in range(1, len(model.predictor)+1):
    print('layer %d' % l)
    y = model.predictor(x, depth=l)
    for i in range(math.ceil(args.num/10)):
        print('.', end="")
        sys.stdout.flush()
        for j in range(10):
            n = i*10 + j
            n1 = (i*2)*10 + j
            n2 = (i*2+1)*10 + j
            draw_digit_ae(x_train[perm[n]], n1, math.ceil(args.num/10)*2, 10)
            draw_digit_ae(y.data[n], n2, math.ceil(args.num/10)*2, 10)
    print()
    plt.savefig('dump_{}.png'.format(l))

