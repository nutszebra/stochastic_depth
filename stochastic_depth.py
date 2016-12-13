import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Conv_BN_ReLU_Conv_BN(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_sizes=(3, 3), strides=(1, 1), pads=(1, 1), probability=1.0):
        super(Conv_BN_ReLU_Conv_BN, self).__init__()
        modules = []
        modules += [('conv1', L.Convolution2D(in_channel, out_channel, filter_sizes[0], strides[0], pads[0]))]
        modules += [('bn1', L.BatchNormalization(out_channel))]
        modules += [('conv2', L.Convolution2D(out_channel, out_channel, filter_sizes[1], strides[1], pads[1]))]
        modules += [('bn2', L.BatchNormalization(out_channel))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.pads = pads
        self.probability = probability

    def _conv_initialization(self, conv):
        conv.W.data = self.weight_relu_initialization(conv)
        conv.b.data = self.bias_initialization(conv, constant=0)

    def weight_initialization(self):
        self._conv_initialization(self.conv1)
        self._conv_initialization(self.conv2)

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 in self.strides:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        if train is True and self.probability <= np.random.rand():
            # do nothing
            return x
        else:
            batch, channel, height, width = x.data.shape
            _, in_channel, _, _ = self.conv1.W.data.shape
            x = self.concatenate_zero_pad(x, (batch, in_channel, height, width), x.volatile, type(x.data))
            h = self.conv1(x)
            h = self.bn1(h, test=not train)
            h = F.relu(h)
            h = self.conv2(h)
            h = self.bn2(h, test=not train)
            # expectation
            if train is False:
                h = h * self.probability
            h = h + self.concatenate_zero_pad(self.maybe_pooling(x), h.data.shape, h.volatile, type(h.data))
            return F.relu(h)

    @staticmethod
    def _count_conv(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        return self._count_conv(self.conv1) + self._count_conv(self.conv2)


class StochasticDepthBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, strides, probability):
        super(StochasticDepthBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        modules = []
        for i in six.moves.range(len(strides)):
            stride = strides[i]
            p = probability[i]
            name = 'res_block{}'.format(i)
            modules.append((name, Conv_BN_ReLU_Conv_BN(in_channel, out_channel, (3, 3), stride, (1, 1), p)))
            # in_channel is changed
            in_channel = out_channel
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.strides = strides
        self.probability = probability

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        for i in six.moves.range(len(self.strides)):
            name = 'res_block{}'.format(i)
            x = self[name](x, train=train)
        return x


class StochasticDepth(nutszebra_chainer.Model):

    def __init__(self, category_num, N=(int(110 / 3 / 2),) * 3, out_channels=(16, 32, 64), p=(1.0, 0.5)):
        super(StochasticDepth, self).__init__()
        # conv
        modules = [('conv1', Conv_BN_ReLU(3, 16, 3, 1, 1))]
        # strides
        strides = [[(1, 1) for _ in six.moves.range(i)] for i in N]
        strides[1][0] = (1, 2)
        strides[2][0] = (1, 2)
        # channels
        drop_probability = StochasticDepth.linear_schedule(p[0], p[1], N)
        in_channel = 16
        for i in six.moves.range(len(strides)):
            out_channel = out_channels[i]
            stride = strides[i]
            probability = drop_probability[i]
            name = 'stochastic_block{}'.format(i)
            modules.append((name, StochasticDepthBlock(in_channel, out_channel, stride, probability)))
            # in_channel is changed
            in_channel = out_channel
        modules += [('linear', Conv_BN_ReLU(out_channel, category_num, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.N = N
        self.out_channels = out_channels
        self.p = p
        self.strides = strides
        self.drop_probability = drop_probability
        self.category_num = category_num
        self.name = 'stochastic_depth_{}_{}_{}_{}'.format(category_num, N, out_channels, p)

    @staticmethod
    def linear_schedule(bottom_layer, top_layer, N):
        total_block = sum(N)

        def y(x):
            return (float(-1 * bottom_layer) + top_layer) / (total_block) * x + bottom_layer
        theta = []
        count = 0
        for num in N:
            tmp = []
            for i in six.moves.range(count, count + num):
                tmp.append(y(i))
            theta.append(tmp)
            count += num
        return theta

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        h = self.conv1(x, train=train)
        for i in six.moves.range(len(self.strides)):
            name = 'stochastic_block{}'.format(i)
            h = self[name](h, train)
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels, 1, 1))
        return F.reshape(self.linear(h, train=train), (batch, self.category_num))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
