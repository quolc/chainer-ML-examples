import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report

class AutoEncoder(chainer.Chain):
    def __init__(self, n_in, n_units, activation):
        super(AutoEncoder, self).__init__(
            l1 = L.Linear(n_in, n_units),
            l2 = L.Linear(n_units, n_in)
        )
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid}[activation]

    def __call__(self, x, train):
        h1 = F.dropout(self.activation(self.l1(x)), train=train)
        return F.dropout(self.l2(h1), train=train)

class Regression(chainer.Chain):
    def __init__(self, predictor):
        super(Regression, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x, True)
        self.loss = F.mean_squared_error(y, t)
        report({'loss': self.loss}, self)
        return self.loss

    def dump(self, x):
        return self.predictor(x, False)

