import time
import sys

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import report, computational_graph
from chainer import optimizers, serializers
from chainer import Chain

from net import Classifier
from net import SimplePerceptron

# シリアライズしたモデルの読み込み
model = Classifier(SimplePerceptron())
serializers.load_npz('linear.model', model)

# 学習したパラメータの表示
print('W')
print(model.predictor.l1.W.data)
print()
print('b')
print(model.predictor.l1.b.data)
print()

while True:
    inp = input()
    if len(inp) == 0:
        break

    a_raw, b_raw = map(lambda x: float(x), inp.split())
    x = chainer.Variable(np.array([[a_raw, b_raw]], dtype="float32"))

    result = F.softmax(model.predictor(x)).data[0]

    if (result[0] > result[1]):
        print ('a > b')
    else:
        print ('a < b')

