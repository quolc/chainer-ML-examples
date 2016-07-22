import argparse
import math
import numpy as np

import chainer
from chainer import serializers

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data
import net
from net import AutoEncoder, Regression

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

parser = argparse.ArgumentParser(description="MNIST Auto-Encoder Tester")
# file loading settings
parser.add_argument('--model', '-m', default='', help='Model of encoder')
parser.add_argument('--num', '-n', default=100, type=int,
                    help='Number of images to dump')
# network structure settings
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--activation', '-a', choices=('relu', 'sigmoid'),
                    default='relu', help="activation function")

args = parser.parse_args()

n_units = args.unit
activation = args.activation

print('MNIST Auto-Encoder Tester')
print()

print('activation func: %s' % activation)
print('# unit: %d' % n_units)

# load dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255

N = data.num_train
x_train, x_test = np.split(mnist['data'], [N])

# initialize model
model = Regression(AutoEncoder(784, n_units, activation))
serializers.load_npz(args.model, model)

# process
perm = np.random.permutation(N)
x = chainer.Variable(np.asarray(x_train[perm[0:args.num]]))
y = model.dump(x)

# dump
print('dumping images')
for i in range(math.ceil(args.num/10)):
    print('row', i+1)
    for j in range(10):
        n = i*10 + j
        n1 = (i*2)*10 + j
        n2 = (i*2+1)*10 + j
        draw_digit_ae(x_train[perm[n]], n1, math.ceil(args.num/10)*2, 10)
        draw_digit_ae(y.data[n], n2, math.ceil(args.num/10)*2, 10)
plt.savefig('dump.png')

