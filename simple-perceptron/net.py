import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import report, computational_graph
from chainer import optimizers, serializers
from chainer import Chain

# (2,2) 単純パーセプトロン
class SimplePerceptron(Chain):
    def __init__(self):
        super(SimplePerceptron, self).__init__(
            l1=L.Linear(2, 2),
        )

    def __call__(self, x):
        return self.l1(x)

# 分類器
class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss

