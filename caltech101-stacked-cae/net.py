import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report

# Deep Convolutional AutoEncoder
# [architecture]
# encoding: input -> (conv (-> pool)) * N -> (FC) * N
# decoding: (FC) * N -> ((unpool) -> deconv) * N -> output
class DCAE(chainer.Chain):
    # conv_arch: [(5, 8, 0, 2), (3, 4, 1, 0)]
    #  -> conv1: (5x5, 8ch, 0pad, 2x2 pooling), conv2: (3x3, 4ch, 1pad, non-pooling)
    # fc_arch: [200, 2]
    #  -> fc1: (output-of-convs -> 200), fc2: (200 -> 2)
    def __init__(self, input_size, conv_arch, fc_arch, activation='relu'):
        super(DCAE, self).__init__(
        )
        self.input_size = input_size
        self.conv_arch = conv_arch
        self.fc_arch = fc_arch
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid}[activation]

        print('[network architecture]')
        print('input: 3x{}x{}'.format(input_size, input_size))

        # layer registration
        n_ch = 3
        dim = input_size
        self.convs = []
        self.deconvs = []
        self.encs = []
        self.decs = []
        self.dims = [input_size]
        for i in range(len(conv_arch)):
            # convolution
            layer = conv_arch[i]
            if dim < layer[0]:
                raise 'too large convolution size'
            conv = L.Convolution2D(n_ch, layer[1], ksize=layer[0], pad=layer[2])
            self.add_link('conv%d' % i, conv)
            self.convs.append(conv)
            print('conv{}: {}x{}x{} (pad{})'.format(
                i, layer[1], layer[0], layer[0], layer[2]))
            dim = dim - (layer[0] - 1) + layer[2] * 2
            print(' -> {}x{}x{}'.format(
                layer[1], dim, dim))
            # deconvolution
            deconv = L.Deconvolution2D(layer[1], n_ch, ksize=layer[0], pad=layer[2])
            self.add_link('deconv%d' % i, deconv)
            self.deconvs.append(deconv)
            # update n_ch and dim
            n_ch = layer[1]
            if layer[3] > 1: # pooling
                if dim % layer[3] != 0:
                    raise 'pooling size non-dividable'
                dim = dim // layer[3]
                print('pool{}: {}x{}'.format(i, layer[3], layer[3]))
                print(' -> {}x{}x{}'.format(layer[1], dim, dim))
            self.dims.append(dim)
        units = dim*dim*n_ch
        for i in range(len(fc_arch)):
            l_enc = L.Linear(units, fc_arch[i])
            self.add_link('enc%d' % i, l_enc)
            self.encs.append(l_enc)
            print('fc{}: {} -> {}'.format(i, units, fc_arch[i]))
            l_dec = L.Linear(fc_arch[i], units)
            self.add_link('dec%d' % i, l_dec)
            self.decs.append(l_dec)
            # update dim
            units = fc_arch[i]
        print()

    def __call__(self, x, train):
        h = x
        # convolution
        for i in range(len(self.convs)):
            h = self.activation(self.convs[i](h))
            if self.conv_arch[i][3] > 1:
                h = F.max_pooling_2d(h, self.conv_arch[i][3], self.conv_arch[i][3])
        # fc enc
        for i in range(len(self.encs)):
            h = F.dropout(self.activation(self.encs[i](h)), train=train)
        # fc dec
        for i in reversed(range(len(self.decs))):
            h = self.activation(self.decs[i](h))
        # deconv
        h = F.reshape(h, (h.data.shape[0], self.conv_arch[-1][1], self.dims[-1], self.dims[-1]))
        for i in reversed(range(len(self.deconvs))):
            if self.conv_arch[i][3] > 1:
                unpool_size = self.dims[i+1] * self.conv_arch[i][3]
                h = F.unpooling_2d(h, self.conv_arch[i][3],
                                   outsize=(unpool_size, unpool_size))
            h = self.activation(self.deconvs[i](h))
        return h

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

