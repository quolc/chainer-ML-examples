import chainer
import chainer.functions as F
import chainer.links as L

# Simplified single-GPU AlexNet without partition toward the channel axis.
class AlexNet(chainer.Chain):
    def __init__(self, activation='relu'):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 101),
        )
        self.train = True

    def __call__(self, x, train=True):
        # 227*227 --conv--> 55*55 --pool--> 27*27
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        # 27*27 --conv--> 27*27 --pool--> 13*13
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        # 13*13 --conv--> 13*13
        h = F.relu(self.conv3(h))
        # 13*13 --conv--> 13*13
        h = F.relu(self.conv4(h))
        # 13*13 --conv--> 13*13 --pool--> 6*6
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        # 6*6*256 = 9216 --FC--> 4096
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        # 4096 --FC--> 4096
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        # 4096 --FC--> 101
        return self.fc8(h)

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

