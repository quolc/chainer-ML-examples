import chainer
import chainer.functions as F
import chainer.links as L
#from chainer import report

class AutoEncoder(chainer.Chain):
    def __init__(self, n_in, n_out, activation='relu', tied=True):
        if tied:
            super(AutoEncoder, self).__init__(
                l1 = L.Linear(n_in, n_out)
            )
            self.add_param('decoder_bias', n_in)
            self.decoder_bias.data[...] = 0
        else:
            super(AutoEncoder, self).__init__(
                l1 = L.Linear(n_in, n_out),
                l2 = L.Linear(n_out, n_in)
            )
        self.tied = tied
        self.n_in = n_in
        self.n_out = n_out
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid}[activation]

    def __call__(self, x, train=True):
        h1 = F.dropout(self.activation(self.l1(x)), train=train)
        if self.tied:
            return F.dropout(F.linear(h1, F.transpose(self.l1.W), self.decoder_bias), train=train)
        else:
            return F.dropout(self.l2(h1), train=train)

    def encode(self, x, train=True):
        return F.dropout(self.activation(self.l1(x)), train=train)

    def decode(self, x, train=True):
        if self.tied:
            return F.dropout(F.linear(x, F.transpose(self.l1.W), self.decoder_bias), train=train)
        else:
            return F.dropout(self.l2(x), train=train)

class StackedAutoEncoder(chainer.ChainList):
    def __init__(self, autoencoders):
        super(StackedAutoEncoder, self).__init__()
        for ae in autoencoders:
            self.add_link(ae)

    def __call__(self, x, train=True, depth=0):
        if depth == 0: depth = len(self)
        h = x
        for i in range(depth):
            h = self[i].encode(h, train=train)
        for i in range(depth):
            h = self[depth-1-i].decode(h, train=train)
        return h

    def encode(self, x, train=True, depth=0):
        if depth == 0: depth = len(self)
        h = x
        for i in range(depth):
            h = self[i].encode(h, train=train)
        return h

    def decode(self, x, train=True, depth=0):
        if depth == 0: depth = len(self)
        h = x
        for i in range(depth):
            h = self[depth-1-i].decode(h, train=train)
        return h

class Regression(chainer.Chain):
    def __init__(self, predictor):
        super(Regression, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x, True)
        self.loss = F.mean_squared_error(y, t)
#        report({'loss': self.loss}, self)
        return self.loss

    def dump(self, x):
        return self.predictor(x, False)

