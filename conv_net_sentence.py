"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
import warnings
import time
import pandas as pd
import sys
warnings.filterwarnings("ignore")


# different non-linearities
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


def train_conv_net(datasets,
                   U,
                   P,
                   filter_hs,
                   hidden_units,
                   dropout_rate,
                   shuffle_batch,
                   n_epochs,
                   batch_size,
                   lr_decay,
                   conv_non_linear,
                   activations,
                   sqr_norm_lim,
                   non_static):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    img_w = U.shape[1] + P.shape[1]  # w2v dim + p2v dim
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0]) - 1) / 2  # seq len
    filter_w = img_w
    feature_maps = hidden_units[0]  # filters
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape", img_h, img_w),
                  ("filter shape", filter_shapes),
                  ("pool sizes", pool_sizes),
                  ("hidden_units", hidden_units),
                  ("dropout", dropout_rate),
                  ("batch_size", batch_size),
                  ("learn_decay", lr_decay),
                  ("conv_non_linear", conv_non_linear),
                  ("non_static", non_static),
                  ("sqr_norm_lim", sqr_norm_lim),
                  ("shuffle_batch", shuffle_batch)]
    print parameters

    ##########################
    #   model architecture   #
    ##########################

    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')  # words
    y = T.ivector('y')
    z = T.matrix('z')  # tags

    Words = theano.shared(value=U, name="Words")
    Tags = theano.shared(value=P, name="Tags")
    layer0_input_words = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, img_h, Words.shape[1]))
    layer0_input_tags = Tags[T.cast(z.flatten(), dtype="int32")].reshape((z.shape[0], 1, img_h, Tags.shape[1]))
    layer0_input = T.concatenate([layer0_input_words, layer0_input_tags], 3)  # TODO: concat words and tags !!!

    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng,
                                        input=layer0_input,
                                        image_shape=(batch_size, 1, img_h, img_w),
                                        filter_shape=filter_shape,
                                        poolsize=pool_size,
                                        non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs, 1)
    hidden_units[0] = feature_maps * len(filter_hs)
    classifier = MLPDropout(rng,
                            input=layer1_input,
                            layer_sizes=hidden_units,
                            activations=activations,
                            dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
        params += [Tags]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    ##########################
    #    dataset handling    #
    ##########################

    # train
    if len(datasets[0]) % batch_size != 0:
        datasets[0] = np.random.permutation(datasets[0])
        to_add = batch_size - len(datasets[0]) % batch_size
        datasets[0] = np.concatenate((datasets[0], datasets[0][:to_add]))
    train_set_x, train_set_y, train_set_z = \
        shared_dataset((datasets[0][:, :img_h], datasets[0][:, -1], datasets[0][:, img_h:2*img_h]))
    n_train_batches = int(len(datasets[0]) / batch_size)

    # val
    if len(datasets[1]) % batch_size != 0:
        datasets[1] = np.random.permutation(datasets[1])
        to_add = batch_size - len(datasets[1]) % batch_size
        datasets[1] = np.concatenate((datasets[1], datasets[1][:to_add]))
    val_set_x, val_set_y, val_set_z = \
        shared_dataset((datasets[1][:, :img_h], datasets[1][:, -1], datasets[1][:, img_h:2*img_h]))
    n_val_batches = int(len(datasets[1]) / batch_size)

    # test
    test_set_x = datasets[2][:, :img_h]
    test_set_z = datasets[2][:, img_h:2*img_h]
    test_set_y = np.asarray(datasets[2][:, -1], "int32")

    ##########################
    #    theano functions    #
    ##########################

    zero_vec_tensor = T.vector()
    set_zero_word = theano.function([zero_vec_tensor],
                               updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))],
                               allow_input_downcast=True)
    set_zero_pos = theano.function([zero_vec_tensor],
                               updates=[(Tags, T.set_subtensor(Tags[0, :], zero_vec_tensor))],
                               allow_input_downcast=True)

    val_model = theano.function([index], classifier.errors(y),
                                givens={
                                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: val_set_y[index * batch_size: (index + 1) * batch_size],
                                    z: val_set_z[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
                                 givens={
                                     x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: train_set_y[index * batch_size: (index + 1) * batch_size],
                                     z: train_set_z[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_set_x[index*batch_size:(index+1)*batch_size],
                                      y: train_set_y[index*batch_size:(index+1)*batch_size],
                                      z: train_set_z[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast=True)

    ##########################
    #  theano test function  #
    ##########################

    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input_words = Words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, img_h, Words.shape[1]))
    test_layer0_input_tags = Tags[T.cast(z.flatten(), dtype="int32")].reshape((test_size, 1, img_h, Tags.shape[1]))
    test_layer0_input = T.concatenate([test_layer0_input_words, test_layer0_input_tags], 3)  # TODO: concat !!
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x, y, z],
                                     test_error,
                                     allow_input_downcast=True)

    ##########################
    #        training        #
    ##########################
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    best_test_perf = 0
    cost_epoch = 0
    best_epoch = 0

    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero_word(np.zeros(U.shape[1]))
                set_zero_pos(np.zeros(P.shape[1]))
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero_word(np.zeros(U.shape[1]))
                set_zero_pos(np.zeros(P.shape[1]))
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1 - np.mean(val_losses)
        test_loss = test_model_all(test_set_x, test_set_y, test_set_z)
        test_perf = 1 - test_loss

        print 'epoch: {}, time: {} secs, train: {}, val: {}, test: {}'\
            .format(epoch, time.time() - start_time, train_perf * 100., val_perf * 100., test_perf * 100.)
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            best_test_perf = test_perf
            best_epoch = epoch
    return best_test_perf, best_epoch


def shared_dataset(data_xyz, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y, data_z = data_xyz
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_z = theano.shared(np.asarray(data_z,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32'), shared_z


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def get_idx_from_sent(sent, word_idx_map, max_l, filter_h):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            print "{} does not exist !!".format(word)
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


# TODO: decide val set HERE !!
# TODO: for sstb, split # [0, 2]
# TODO: for trec, split # [0, 1]
def make_idx_data_mr(revs, word_idx_map, pos_idx_map, cv, max_l, filter_h, val_ratio=0.1):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainval, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        sent.extend(get_idx_from_sent(rev["tag"], pos_idx_map, max_l, filter_h))
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)        
        else:  
            trainval.append(sent)
    trainval = np.array(trainval, dtype="int")
    test = np.array(test, dtype="int")

    trainval = np.random.permutation(trainval)
    val_size = int(len(trainval) * val_ratio)
    val = trainval[:val_size]
    train = trainval[val_size:]

    return [train, val, test]
  
   
def make_idx_data_trec(revs, word_idx_map, pos_idx_map, max_l, filter_h, val_ratio=0.1):
    trainval, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        sent.extend(get_idx_from_sent(rev["tag"], pos_idx_map, max_l, filter_h))
        sent.append(rev["y"])
        if rev["split"] == 0:
            trainval.append(sent)
        else:
            test.append(sent)
    trainval = np.array(trainval, dtype="int")
    test = np.array(test, dtype="int")

    trainval = np.random.permutation(trainval)
    val_size = int(len(trainval) * val_ratio)
    val = trainval[:val_size]
    train = trainval[val_size:]

    return [train, val, test]


def make_idx_data_sstb(revs, word_idx_map, pos_idx_map, max_l, filter_h):
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        sent.extend(get_idx_from_sent(rev["tag"], pos_idx_map, max_l, filter_h))
        sent.append(rev["y"])
        if rev["split"] == 0:
            train.append(sent)
        elif rev["split"] == 1:
            test.append(sent)
        else:
            val.append(sent)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    val = np.array(val, dtype="int")
    return [train, val, test]


if __name__=="__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'sstb'
    print "loading data...{}".format(dataset),
    if dataset == 'trec':
        x = cPickle.load(open("trec.p", "rb"))
    elif dataset == 'mr':
        x = cPickle.load(open("mr.p", "rb"))
    elif dataset == 'sstb':
        x = cPickle.load(open("sstb.p", "rb"))
    else:
        print "invalid dataset"
        sys.exit()

    revs, W, W_rand, word_idx_map, vocab, P, P_rand, pos_idx_map, num_folds, num_classes = x  # TODO: get K HERE !!
    print "data loaded!"
    non_static = True
    execfile("conv_net_classes.py")
    results = []
    r = range(0, num_folds)

    W_dim = W.shape[1]
    P_dim = P.shape[1]
    max_len = np.max(pd.DataFrame(revs)["num_words"])

    for i in r:
        if dataset == 'trec':
            datasets = make_idx_data_trec(revs, word_idx_map, pos_idx_map, max_l=max_len, filter_h=5)
        elif dataset == 'mr':
            datasets = make_idx_data_mr(revs, word_idx_map, pos_idx_map, cv=i, max_l=max_len, filter_h=5)
        elif dataset == 'sstb':
            datasets = make_idx_data_sstb(revs, word_idx_map, pos_idx_map, max_l=max_len, filter_h=5)
        print "Train/Val/Test set: {}/{}/{}".format(len(datasets[0]), len(datasets[1]), len(datasets[2]))
        perf, epoch = train_conv_net(datasets,
                                     W,
                                     P_rand,
                                     filter_hs=[3, 4, 5],
                                     hidden_units=[100, num_classes],
                                     dropout_rate=[0.5],
                                     shuffle_batch=True,
                                     n_epochs=100,
                                     batch_size=50,
                                     lr_decay=0.95,
                                     conv_non_linear="relu",
                                     activations=[Iden],
                                     sqr_norm_lim=9,
                                     non_static=non_static)
        print "cv: {}, perf: {}, epoch: {}".format(i, perf, epoch)
        results.append(perf)
    print str(np.mean(results))
