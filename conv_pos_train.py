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
import argparse
from conv_net_classes import MLPDropout, LeNetConvPoolLayer
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


def train_pos_cnn(datasets,
                  W,
                  P,
                  filter_hs,
                  hidden_units,
                  dropout_rate,
                  n_epochs,
                  batch_size,
                  lr_decay,
                  conv_non_linear,
                  activations,
                  sqr_norm_lim,
                  model):

    # print params
    parameters = [("num_filters", hidden_units[0]),
                  ("num_classes", hidden_units[1]),
                  ("filter_types", filter_hs),
                  ("dropout", dropout_rate),
                  ("num_epochs", n_epochs),
                  ("batch_size", batch_size),
                  ("learn_decay", lr_decay),
                  ("conv_non_linear", conv_non_linear),
                  ("sqr_norm_lim", sqr_norm_lim),
                  ("model", model)]
    print parameters

    ##########################
    #   model architecture   #
    ##########################

    print 'building the model architecture...'
    index = T.lscalar()
    x = T.matrix('x')  # words
    y = T.ivector('y')  # labels
    z = T.matrix('z')  # tags
    curr_batch_size = T.lscalar()

    # set necessary variables
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0]) - 1) / 2  # input height = seq len
    feature_maps = hidden_units[0]  # num filters

    # EMBEDDING LAYER
    Words = theano.shared(value=W, name="Words")
    Tags = theano.shared(value=P, name="Tags")
    emb_layer_params = [Words] + [Tags]

    if model == "concat":
        print 'use concat...'
        layer0_input_words = Words[T.cast(x.flatten(), dtype="int32")].reshape((curr_batch_size, 1, img_h, Words.shape[1]))
        layer0_input_tags = Tags[T.cast(z.flatten(), dtype="int32")].reshape((curr_batch_size, 1, img_h, Tags.shape[1]))
        layer0_input = T.concatenate([layer0_input_words, layer0_input_tags], 3)  # curr_batch_size, 1, img_h, D+M
        img_w = W.shape[1] + P.shape[1]

    elif model == "mult":
        print 'use mult...'
        window = 5  # TODO: set window size !!!
        CW = Words[T.cast(x.flatten(), dtype="int32")].reshape((curr_batch_size, 1, img_h, Words.shape[1]))
        CP = Tags[T.cast(z.flatten(), dtype="int32")].reshape((curr_batch_size, img_h, Tags.shape[1]))
        left_pad = CP[:, CP.shape[1]-int(window/2):, :]
        right_pad = CP[:, :int(window/2), :]
        CP_padded = T.concatenate([left_pad, CP, right_pad], 1)
        CP_stack_list = []
        for k in range(window):
            CP_stack_list.append(CP_padded[:, k:k+img_h, :])
        CP_stack = T.concatenate(CP_stack_list, 2)
        CP_stack = CP_stack.reshape((curr_batch_size, 1, img_h, Tags.shape[1]*window))
        CW_CP = T.concatenate([CW, CP_stack], 3)
        F = W.shape[1]  # TODO: set F, final dim to represent each token !!!
        Q = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                 high=0.01,
                                                 size=[W.shape[1] + P.shape[1]*window, F]),
                                     dtype=theano.config.floatX),
                          borrow=True,
                          name="Q")
        emb_layer_params += [Q]  # add Q to the list of params to train
        layer0_input = ReLU(T.dot(CW_CP, Q))  # curr_batch_size, 1, img_h, F
        img_w = F

    else:
        print "invalid model"
        sys.exit()

    # set more variables
    filter_w = img_w  # filter width = input matrix width

    # construct filter shapes and pool sizes
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

    # CONV-POOL LAYER
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_shapes)):
        conv_layer = LeNetConvPoolLayer(rng,
                                        input=layer0_input,
                                        image_shape=(None, 1, img_h, img_w),
                                        filter_shape=filter_shapes[i],
                                        poolsize=pool_sizes[i],
                                        non_linear=conv_non_linear)
        layer1_inputs.append(conv_layer.output.flatten(2))
        conv_layers.append(conv_layer)
    layer1_input = T.concatenate(layer1_inputs, 1)
    hidden_units[0] = feature_maps * len(filter_shapes)  # update the hidden units

    # OUTPUT LAYER (Dropout, Fully-Connected, Soft-Max)
    classifier = MLPDropout(rng,
                            input=layer1_input,
                            layer_sizes=hidden_units,
                            activations=activations,
                            dropout_rates=dropout_rate)

    # UPDATE
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    params += emb_layer_params
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)  # use this to update
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    ##########################
    #    dataset handling    #
    ##########################

    print 'handling dataset...'
    # train
    # if len(datasets[0]) % batch_size != 0:
    #     datasets[0] = np.random.permutation(datasets[0])
    #     to_add = batch_size - len(datasets[0]) % batch_size
    #     datasets[0] = np.concatenate((datasets[0], datasets[0][:to_add]))
    train_set_x, train_set_y, train_set_z = \
        shared_dataset((datasets[0][:, :img_h], datasets[0][:, -1], datasets[0][:, img_h:2*img_h]))
    n_train_batches = int(len(datasets[0]) / batch_size)

    # val
    # if len(datasets[1]) % batch_size != 0:
    #     datasets[1] = np.random.permutation(datasets[1])
    #     to_add = batch_size - len(datasets[1]) % batch_size
    #     datasets[1] = np.concatenate((datasets[1], datasets[1][:to_add]))
    val_set_x, val_set_y, val_set_z = \
        shared_dataset((datasets[1][:, :img_h], datasets[1][:, -1], datasets[1][:, img_h:2*img_h]))
    n_val_batches = int(len(datasets[1]) / batch_size)

    # test
    test_set_x, test_set_y, test_set_z = \
        shared_dataset((datasets[2][:, :img_h], datasets[2][:, -1], datasets[2][:, img_h:2*img_h]))
    n_test_batches = int(len(datasets[2]) / batch_size)  # TODO: left-overs
    if len(datasets[2]) % batch_size > 0:
        n_test_batches += 1

    ##########################
    #    theano functions    #
    ##########################

    print 'preparing theano functions...'
    zero_vec_tensor = T.vector()
    set_zero_word = theano.function([zero_vec_tensor],
                                    updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))],
                                    allow_input_downcast=True)
    set_zero_pos = theano.function([zero_vec_tensor],
                                   updates=[(Tags, T.set_subtensor(Tags[0, :], zero_vec_tensor))],
                                   allow_input_downcast=True)
    val_model = theano.function([index, curr_batch_size], classifier.errors(y),
                                givens={
                                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: val_set_y[index * batch_size: (index + 1) * batch_size],
                                    z: val_set_z[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
    train_eval_model = theano.function([index, curr_batch_size], classifier.errors(y),
                                 givens={
                                     x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: train_set_y[index * batch_size: (index + 1) * batch_size],
                                     z: train_set_z[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)
    train_model = theano.function([index, curr_batch_size], cost, updates=grad_updates,
                                  givens={
                                      x: train_set_x[index*batch_size:(index+1)*batch_size],
                                      y: train_set_y[index*batch_size:(index+1)*batch_size],
                                      z: train_set_z[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast=True)
    test_model = theano.function([index, curr_batch_size], classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_set_y[index * batch_size: (index + 1) * batch_size],
                                     z: test_set_z[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)

    ##########################
    #        training        #
    ##########################

    print 'training...'
    epoch = 0
    best_val_perf = 0
    best_test_perf = 0
    best_epoch = 0

    while epoch < n_epochs:
        start_time = time.time()
        epoch += 1
        step = 1
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            cost = train_model(minibatch_index, min(batch_size, len(datasets[0])-minibatch_index*batch_size))
            set_zero_word(np.zeros(W.shape[1]))
            set_zero_pos(np.zeros(P.shape[1]))
            step += 1
        train_losses = [train_eval_model(i, min(batch_size, len(datasets[0])-i*batch_size)) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i, min(batch_size, len(datasets[1])-i*batch_size)) for i in xrange(n_val_batches)]
        val_perf = 1 - np.mean(val_losses)
        test_losses = [test_model(i, min(batch_size, len(datasets[2])-i*batch_size)) for i in xrange(n_test_batches)]
        test_loss = np.mean(test_losses)
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
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        shared_z = theano.shared(np.asarray(data_z, dtype=theano.config.floatX), borrow=borrow)
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
    for word in sent.split():
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


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
    trainval = np.random.permutation(np.array(trainval, dtype="int"))
    test = np.array(test, dtype="int")
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
    trainval = np.random.permutation(np.array(trainval, dtype="int"))
    test = np.array(test, dtype="int")
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


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='trec',
                        help='which dataset to use')
    parser.add_argument('--model', type=str, default='concat',
                        help='which model to use')
    parser.add_argument('--num_repetitions', type=int, default=1,
                        help="how many times to run (for datasets that don't use k folds)")
    parser.add_argument('--num_epochs', type=int, default=25,
                        help="how many epochs")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    # get command line args
    args = get_command_line_args()

    # load data
    print "loading data...{}".format(args.dataset),
    if args.dataset == 'trec':
        x = cPickle.load(open("trec.p", "rb"))
    elif args.dataset == 'mr':
        x = cPickle.load(open("mr.p", "rb"))
    elif args.dataset == 'sstb':
        x = cPickle.load(open("sstb.p", "rb"))
    else:
        print "invalid dataset"
        sys.exit()
    revs, W, W_rand, word_idx_map, vocab, P, P_rand, pos_idx_map, num_folds, num_classes = x
    max_len = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"

    # import conv net classes
    execfile("conv_net_classes.py")

    # start training
    num_folds = args.num_repetitions if num_folds == 1 else num_folds
    test_results = []
    for i in range(num_folds):
        if args.dataset == 'trec':
            datasets = make_idx_data_trec(revs, word_idx_map, pos_idx_map, max_l=max_len, filter_h=5)
        elif args.dataset == 'mr':
            datasets = make_idx_data_mr(revs, word_idx_map, pos_idx_map, cv=i, max_l=max_len, filter_h=5)
        elif args.dataset == 'sstb':
            datasets = make_idx_data_sstb(revs, word_idx_map, pos_idx_map, max_l=max_len, filter_h=5)
        print "train/val/test set: {}/{}/{}".format(len(datasets[0]), len(datasets[1]), len(datasets[2]))
        best_test, best_epoch = train_pos_cnn(datasets,
                                              W,  # use pre-trained word embeddings
                                              P,  # use pre-trained pos embeddings
                                              filter_hs=[3, 4, 5],
                                              hidden_units=[100, num_classes],
                                              dropout_rate=[0.5],
                                              n_epochs=args.num_epochs,
                                              batch_size=50,
                                              lr_decay=0.95,
                                              conv_non_linear="relu",
                                              activations=[Iden],
                                              sqr_norm_lim=9,
                                              model=args.model)
        print "cv: {}, perf: {}, epoch: {}".format(i, best_test, best_epoch)
        test_results.append(best_test)
    print "final perf: {}".format(np.mean(test_results))
