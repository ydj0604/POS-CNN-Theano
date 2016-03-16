"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = T.switch(mask, layer, 0)
    # output = layer * T.cast(mask, theano.config.floatX)
    return output


def dropout(rng, x, p, is_train):
    masked_x = None
    if p > 0.0 and p < 1.0:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0-p,
            size=x.shape,
            dtype=theano.config.floatX
        )
        # mask *= 1.0 / (1.0 - p)
        # masked_x = x * mask
        masked_x = T.switch(mask, x, 0)
    else:
        masked_x = x
    return T.switch(T.neq(is_train, 0), masked_x, x*(1.0-p))


class MLPDropout(object):
    def __init__(self, rng, input, weight_matrix_shape, dropout_rate, use_bias=True):
        # inputs
        next_layer_input = input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rate)

        # Set up the output layer
        n_in, n_out = weight_matrix_shape
        self.dropout_output_layer = LogisticRegression(input=next_dropout_layer_input, n_in=n_in, n_out=n_out)

        # Again, reuse paramters in the dropout output.
        self.output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=self.dropout_output_layer.W * (1 - dropout_rate),
            b=self.dropout_output_layer.b,
            n_in=n_in, n_out=n_out)

        # Use the negative log likelihood of the logistic regression layer as the objective.
        self.dropout_negative_log_likelihood = self.dropout_output_layer.negative_log_likelihood
        self.dropout_errors = self.dropout_output_layer.errors

        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

        # Grab all the parameters together.
        self.params = [param for param in self.dropout_output_layer.params]

    def predict(self, new_input):
        p_y_given_x = T.nnet.softmax(T.dot(new_input, self.output_layer.W) + self.output_layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b')
        else:
            self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize, non_linear):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))

        # initialize weights with random weights
        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True, name="W_conv")
        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True,name="W_conv")
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear == "relu":
            conv_out_relu = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_relu, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]
        
    def predict(self, new_data, batch_size):
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear == "relu":
            conv_out_relu = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_relu, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output
