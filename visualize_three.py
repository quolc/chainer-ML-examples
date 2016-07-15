import time
import sys

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import report, computational_graph
from chainer import optimizers, serializers
from chainer import Chain

import matplotlib.pyplot as plt

from net import Classifier
from net import ThreePerceptron

n_unit = 300

# シリアライズしたモデルの読み込み
model = Classifier(ThreePerceptron(n_unit))
serializers.load_npz('linear.model', model)

# 学習したパラメータの表示
print('W')
print(model.predictor.l1.W.data)
print()
print('b')
print(model.predictor.l1.b.data)
print()

X, Y = np.mgrid[-1:1:101j, -1:1:101j]
z_raw = []

for i in range(0, 101):
    row = []
    z_raw.append(row)
    for j in range(0, 101):
        x = X[i][j]
        y = Y[i][j]
        v = chainer.Variable(np.array([[x, y]], dtype="float32"))
        result = F.softmax(model.predictor(v)).data[0]

        row.append(0.0 * result[0] + 1.0 * result[1])

Z = np.array(z_raw, dtype="float32")
plt.pcolor(X, Y, Z, cmap='RdBu', vmin=0, vmax=1)
plt.colorbar()

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.axis('tight')
plt.axes().set_aspect('equal')

filename = "output.png"
plt.savefig(filename)

