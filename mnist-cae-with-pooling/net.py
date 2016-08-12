import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report

# Simple Convolutional AutoEncoder
# [structure (filter_size = 9, n_units = 100)]
# (encoding)
# - input   28*28*1
# - conv1   28*28*1 -> 20*20*8 (filter 9*9*8)
# - pool1   20*20*8 -> 10*10*8 (max-pooling 2*2)
# - fc1     800 -> 100
# (decoding)
# - fc2     100 -> 800
# - unpool1 10*10*8 -> 20*20*8
# - deconv1 20*20*8 -> 28*28*1 (filter 9*9*8)
class AutoEncoder(chainer.Chain):
    def __init__(self, input_size, n_filters=10, n_units=20, filter_size=9, activation='relu'):
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid}[activation]
        self.n_filters = n_filters
        self.n_units = n_units
        self.filter_size = filter_size
        self.dim1 = (input_size - filter_size + 1)
        self.dim2 = self.dim1 // 2

        super(AutoEncoder, self).__init__(
            conv1 = L.Convolution2D(1, n_filters, filter_size),
            lenc1 = L.Linear(n_filters*self.dim2*self.dim2, n_units),
            ldec1 = L.Linear(n_units, n_filters*self.dim2*self.dim2),
            deconv1 = L.Deconvolution2D(n_filters, 1, filter_size)
        )

    def __call__(self, x, train):
        # conv1
        h1 = self.activation(self.conv1(x))
        # pool1
        h2 = F.max_pooling_2d(h1, 2, 2)
        # fc1
        h3 = F.dropout(self.activation(self.lenc1(h2)), train=train)
        # fc2
        h4 = F.reshape(self.activation(self.ldec1(h3)),
                      (x.data.shape[0], self.n_filters, self.dim2, self.dim2))
        # unpool1
        h5 = F.unpooling_2d(h4, 2, outsize=(self.dim1, self.dim1))
        # unconv1
        h6 = self.activation(self.deconv1(h5))
        return h6

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

