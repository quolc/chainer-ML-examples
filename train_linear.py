# 組み込みのClassifierやTrainerを使わず，単純パーセプトロンを最適化する
import time
import sys

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import report, computational_graph
from chainer import optimizers, serializers
from chainer import Chain

from net import SimplePerceptron
from net import Classifier

batchsize = 100
n_epoch = 100

# training data size
N = int(input())

# データセットを標準入力から読み込む
train_data_raw = []
train_target_raw = []
for i in range(0, N):
    i0, i1, t = input().split()
    train_data_raw.append([i0, i1])
    train_target_raw.append(t)
# NumPy配列に変換
train_data = np.array(train_data_raw, dtype="float32")
train_target = np.array(train_target_raw, dtype="int32")

# (2,2) 単純パーセプトロン
class SimplePerceptron(Chain):
    def __init__(self):
        super(SimplePerceptron, self).__init__(
            l1=L.Linear(2, 2),
        )

    def __call__(self, x):
        return F.relu(self.l1(x))

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

# モデルインスタンスの作成
model = Classifier(SimplePerceptron())

# オプティマイザの初期化
optimizer = optimizers.Adam()
optimizer.setup(model)

# トレーニングループ
for epoch in range(0, n_epoch):
    print ('epoch', epoch+1)

    # ランダムに訓練データを並べ替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0

    start = time.time()
    for i in range(0, N, batchsize):
        # データを取り出し
        x = chainer.Variable(np.asarray(train_data[perm[i:i + batchsize]]))
        t = chainer.Variable(np.asarray(train_target[perm[i:i + batchsize]]))

        # オプティマイズ
        optimizer.update(model, x, t)

        # グラフ出力（一回だけ）
        if epoch == 0 and i == 0:
            with open('graph.dot', 'w') as o:
                variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
                                  'style': 'filled'}
                function_style = {'shape': 'record', 'fillcolor': '#6495ED',
                                  'style': 'filled'}
                g = computational_graph.build_computational_graph(
                    (model.loss, ),
                    variable_style=variable_style,
                    function_style=function_style)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time

    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

print('save the model') 
serializers.save_npz('linear.model', model)

