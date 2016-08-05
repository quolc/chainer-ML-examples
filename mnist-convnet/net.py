import chainer
import chainer.functions as F
import chainer.links as L

# 1-conv 1-hidden convolutional network
#   input: 28px * 28px * 1ch
#   output: 10 units (label: 0-9)
#
#   structure: CONV -> RELU -> POOL -> FC1 -> FC2
#     - CONV: 28px * 28px * 1ch -> 26px * 26px * 8ch
#             3*3 filter, 0 padding
#     - FC1:  1352units -> 256units
#     - FC2:  256units -> 10units
# 
class ConvNet(chainer.Chain):
    def __init__(self, dim_x, dim_y, n_ch_in, n_ch_conv, n_unit1, n_out,
                 ksize=3, activation='relu'):
        padding_loss = ksize-1
        dim_x_conv = dim_x - padding_loss
        dim_y_conv = dim_y - padding_loss

        super(ConvNet, self).__init__(
            conv1 = L.Convolution2D(
                in_channels=n_ch_in, out_channels=n_ch_conv,
                ksize=ksize, pad=0),
            l1 = L.Linear(
                (dim_x_conv//2) * (dim_y_conv//2) * n_ch_conv,
                n_unit1),
            l2 = L.Linear(n_unit1, n_out)
        )
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid, 'identity': F.identity}[activation]

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(self.activation(self.conv1(x)), 2)
        h2 = F.dropout(self.activation(self.l1(h1)), train=train)
        return self.activation(self.l2(h2))

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

