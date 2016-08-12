import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report

# Simple Convolutional AutoEncoder
# [structure (filter_size = 9, n_units = 100)]
# (encoding)
# - input   28*28*1
# - conv1   28*28*1 -> 20*20*8 (filter 9*9*8)
# - fc1     3200 -> 100
# (decoding)
# - fc2     100 -> 3200
# - deconv1 20*20*8 -> 28*28*1 (filter 9*9*8)
class AutoEncoder(chainer.Chain):
    def __init__(self, input_size, n_filters=10, n_units=20, filter_size=9, activation='relu'):
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid}[activation]
        self.n_filters = n_filters
        self.n_units = n_units
        self.dim1 = input_size - filter_size + 1

        super(AutoEncoder, self).__init__(
            conv1 = L.Convolution2D(1, n_filters, filter_size),
            lenc1 = L.Linear(n_filters*self.dim1*self.dim1, n_units),
            ldec1 = L.Linear(n_units, n_filters*self.dim1*self.dim1),
            deconv1 = L.Deconvolution2D(n_filters, 1, filter_size)
        )

    def __call__(self, x, train):
        h1 = self.activation(self.conv1(x))
        h2 = F.dropout(self.activation(self.lenc1(h1)), train=train)
        h3 = F.reshape(self.activation(self.ldec1(h2)), (x.data.shape[0], self.n_filters, self.dim1, self.dim1))
        h4 = self.activation(self.deconv1(h3))
        return h4

class Regression(chainer.Chain):
    def __init__(self, predictor):
        super(Regression, self).__init__(predictor=predictor)

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train)
        self.loss = F.mean_squared_error(y, t)
        report({'loss': self.loss}, self)
        return self.loss

    def dump(self, x):
        return self.predictor(x, False)

