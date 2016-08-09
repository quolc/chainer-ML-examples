import chainer
import chainer.functions as F
import chainer.links as L

# 1-conv 1-hidden convolutional network
#   input: 32px * 32px * 3ch
#   output: 10 units (label: 0-9)
#
#   structure: CONV -> RELU -> CONV -> RELU -> POOL -> FC1 -> FC2
#     - CONV: 32px * 32px * 3ch -> 32px * 32px * 32ch
#             3*3 filter, 1 padding
#     - FC1:  1352units -> 512units
#     - FC2:  512units -> 10units
# 
class ConvNet(chainer.Chain):
    def __init__(self, dim_x, dim_y, n_ch_in, n_ch_conv, n_unit1, n_out,
                 ksize=3, activation='relu'):
        super(ConvNet, self).__init__(
            conv1 = L.Convolution2D(
                n_ch_in, n_ch_conv, ksize=ksize, pad=(ksize-1)//2),
            conv2 = L.Convolution2D(
                n_ch_conv, n_ch_conv, ksize=ksize, pad=(ksize-1)//2),
            conv3 = L.Convolution2D(
                n_ch_conv, n_ch_conv, ksize=ksize, pad=(ksize-1)//2),
            conv4 = L.Convolution2D(
                n_ch_conv, n_ch_conv, ksize=ksize, pad=(ksize-1)//2),
            conv5 = L.Convolution2D(
                n_ch_conv, n_ch_conv, ksize=ksize, pad=(ksize-1)//2),
            conv6 = L.Convolution2D(
                n_ch_conv, n_ch_conv, ksize=ksize, pad=(ksize-1)//2),
            l1 = L.Linear(
                (dim_x//8) * (dim_y//8) * n_ch_conv,
                n_unit1),
            l2 = L.Linear(n_unit1, n_out)
        )
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid, 'identity': F.identity}[activation]

    def __call__(self, x, train=True):
        h1 = self.activation(self.conv1(x))
        h2 = F.max_pooling_2d(self.activation(self.conv2(h1)), 2)
        h3 = self.activation(self.conv3(h2))
        h4 = F.max_pooling_2d(self.activation(self.conv4(h3)), 2)
        h5 = self.activation(self.conv4(h4))
        h6 = F.max_pooling_2d(self.activation(self.conv5(h5)), 2)
        h7 = F.dropout(self.activation(self.l1(h6)), train=train)
        return self.activation(self.l2(h7))

class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(
            predictor = predictor
        )

    def __call__(self, x, t, train=True):
        y = self.predictor(x, train)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

