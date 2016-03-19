import numpy as np
import theano
import theano.tensor as T
import sys

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


def dropout(rng, x, p, is_train):
    masked_x = None
    if p > 0.0 and p < 1.0:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(
            n=1,
            p=1.0-p,
            size=x.shape
        )
        mask *= 1.0 / (1.0 - p)
        masked_x = T.switch(mask, x, 0)
    else:
        masked_x = x
    return T.switch(T.neq(is_train, 0), masked_x, x*(1.0-p))


class EmbeddingLayer(object):
    """A multilayer perceptron with dropout"""
    def __init__(self, rng, is_train, x, z, batch_size, img_h, W, P, model, dropout_rate, window=5):
        # img_h = seq len

        if model == 'notag':
            print 'no tags...'
            self.Words = theano.shared(value=W, name="Words")
            self.params = [self.Words]
            self.output = self.Words[T.cast(x.flatten(), dtype="int32")]\
                .reshape((batch_size, 1, img_h, self.Words.shape[1]))
            self.final_token_dim = W.shape[1]

        if model == 'concat':
            print 'use concat...'
            self.Words = theano.shared(value=W, name="Words")
            self.Tags = theano.shared(value=P, name="Tags")
            self.params = [self.Words, self.Tags]
            layer0_input_words = self.Words[T.cast(x.flatten(), dtype="int32")]\
                .reshape((batch_size, 1, img_h, self.Words.shape[1]))
            layer0_input_tags = self.Tags[T.cast(z.flatten(), dtype="int32")]\
                .reshape((batch_size, 1, img_h, self.Tags.shape[1]))
            output = T.concatenate([layer0_input_words, layer0_input_tags], 3)  # curr_batch_size, 1, img_h, D+M
            self.output = dropout(rng, output, dropout_rate, is_train)
            self.final_token_dim = W.shape[1] + P.shape[1]

        elif model == 'mult':
            print 'use mult...'
            self.Words = theano.shared(value=W, name="Words")
            self.Tags = theano.shared(value=P, name="Tags")
            self.params = [self.Words, self.Tags]
            CW = self.Words[T.cast(x.flatten(), dtype="int32")].reshape((batch_size, 1, img_h, self.Words.shape[1]))
            CP = self.Tags[T.cast(z.flatten(), dtype="int32")].reshape((batch_size, img_h, self.Tags.shape[1]))
            left_pad = CP[:, CP.shape[1]-int(window/2):, :]
            right_pad = CP[:, :int(window/2), :]
            CP_padded = T.concatenate([left_pad, CP, right_pad], 1)  # batch, seq+w, M
            CP_stack_list = []
            for k in range(window):
                CP_stack_list.append(CP_padded[:, k:k+img_h, :])  # batch, seq, M
            CP_stack = T.concatenate(CP_stack_list, 2)  # batch, seq, M*w
            CP_stack = CP_stack.reshape((batch_size, 1, img_h, self.Tags.shape[1]*window))  # batch, 1, seq, M*w
            CW_CP = T.concatenate([CW, CP_stack], 3)  # batch, 1, seqlen, w*M+D
            F = W.shape[1] + P.shape[1]  # TODO: set F, final dim to represent each token !!!
            Q = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                     high=0.01,
                                                     size=[W.shape[1] + P.shape[1]*window, F]),
                                         dtype=theano.config.floatX),
                              borrow=True,
                              name="Q")

            output = ReLU(T.dot(CW_CP, Q))  # curr_batch_size, 1, img_h, F
            self.output = dropout(rng, output, dropout_rate, is_train)
            self.params += [Q]
            self.final_token_dim = F

        elif model == 'tensor':
            print 'use tensor...'
            # first term
            self.Words = theano.shared(value=W, name="Words")
            self.Tags = theano.shared(value=P, name="Tags")
            self.params = [self.Words, self.Tags]
            words = self.Words[T.cast(x.flatten(), dtype="int32")].reshape((batch_size*img_h, self.Words.shape[1]))
            tags = self.Tags[T.cast(z.flatten(), dtype="int32")].reshape((batch_size*img_h, self.Tags.shape[1]))
            words_tags = T.concatenate([words, tags], 1)  # batch * seqlen, D+M
            tags_words = T.concatenate([tags, words], 1)  # batch * seqlen, D+M
            concat_dim = W.shape[1] + P.shape[1]
            F = 50  # TODO: set final dim for each token
            V = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                     high=0.01,
                                                     size=[concat_dim, F, concat_dim]),  # D+M, F, D+M
                                         dtype=theano.config.floatX),
                              borrow=True,
                              name="V")
            words_tags_V = T.tensordot(words_tags, V, [[1], [0]])  # batch * seqlen, F, D+M
            mix_vec = T.batched_dot(words_tags_V, tags_words)  # batch * seqlen, F

            # second term
            Q = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                     high=0.01,
                                                     size=[concat_dim, F]),
                                         dtype=theano.config.floatX),
                              borrow=True,
                              name="Q")

            merge_vec = T.dot(words_tags, Q)  # batch * seqlen, F
            output = ReLU(mix_vec + merge_vec).reshape((batch_size, 1, img_h, F))
            self.output = dropout(rng, output, dropout_rate, is_train)
            self.params += [V, Q]
            self.final_token_dim = F

        else:
            sys.exit()


 # if model == "concat":
 #        print 'use concat...'
 #        layer0_input_words = Words[T.cast(x.flatten(), dtype="int32")].reshape((curr_batch_size, 1, img_h, Words.shape[1]))
 #        layer0_input_tags = Tags[T.cast(z.flatten(), dtype="int32")].reshape((curr_batch_size, 1, img_h, Tags.shape[1]))
 #        layer0_input = T.concatenate([layer0_input_words, layer0_input_tags], 3)  # curr_batch_size, 1, img_h, D+M
 #        img_w = W.shape[1] + P.shape[1]
 #
 #    elif model == "mult":
 #        print 'use mult...'
 #        window = 5  # TODO: set window size !!!
 #        CW = Words[T.cast(x.flatten(), dtype="int32")].reshape((curr_batch_size, 1, img_h, Words.shape[1]))
 #        CP = Tags[T.cast(z.flatten(), dtype="int32")].reshape((curr_batch_size, img_h, Tags.shape[1]))
 #        left_pad = CP[:, CP.shape[1]-int(window/2):, :]
 #        right_pad = CP[:, :int(window/2), :]
 #        CP_padded = T.concatenate([left_pad, CP, right_pad], 1)  # batch, seq+w, M
 #        CP_stack_list = []
 #        for k in range(window):
 #            CP_stack_list.append(CP_padded[:, k:k+img_h, :])  # batch, seq, M
 #        CP_stack = T.concatenate(CP_stack_list, 2)  # batch, seq, M*w
 #        CP_stack = CP_stack.reshape((curr_batch_size, 1, img_h, Tags.shape[1]*window))  # batch, 1, seq, M*w
 #        CW_CP = T.concatenate([CW, CP_stack], 3)  # batch, 1, seqlen, w*M+D
 #        F = W.shape[1] + P.shape[1]  # TODO: set F, final dim to represent each token !!!
 #        Q = theano.shared(np.asarray(rng.uniform(low=-0.01,
 #                                                 high=0.01,
 #                                                 size=[W.shape[1] + P.shape[1]*window, F]),
 #                                     dtype=theano.config.floatX),
 #                          borrow=True,
 #                          name="Q")
 #        emb_layer_params += [Q]  # add Q to the list of params to train
 #        layer0_input = ReLU(T.dot(CW_CP, Q))  # curr_batch_size, 1, img_h, F
 #        img_w = F
 #
 #    elif model == "tensor_nbr":
 #        print 'use tensor...'
 #        window = 5  # TODO: set window size !!!
 #        CW = Words[T.cast(x.flatten(), dtype="int32")].reshape((curr_batch_size, img_h, Words.shape[1]))
 #        CP = Tags[T.cast(z.flatten(), dtype="int32")].reshape((curr_batch_size, img_h, Tags.shape[1]))
 #        left_pad = CP[:, CP.shape[1]-int(window/2):, :]
 #        right_pad = CP[:, :int(window/2), :]
 #        CP_padded = T.concatenate([left_pad, CP, right_pad], 1)  # batch, seq+w, M
 #        CP_stack_list = []
 #        for k in range(window):
 #            CP_stack_list.append(CP_padded[:, k:k+img_h, :])  # batch, seq, M
 #        CP_stack = T.concatenate(CP_stack_list, 2)  # batch, seqlen, w*M
 #
 #        # first term
 #        CW_CP = T.concatenate([CW, CP_stack], 2)  # batch, seqlen, w*M+D
 #        concat_dim = W.shape[1] + P.shape[1]*window
 #        F = 125  # TODO: set F; final dim for each token !!!
 #        V = theano.shared(np.asarray(rng.uniform(low=-0.01,
 #                                                 high=0.01,
 #                                                 size=[concat_dim, F, concat_dim]),
 #                                     dtype=theano.config.floatX),
 #                          borrow=True,
 #                          name="V")
 #        emb_layer_params += [V]  # add Q to the list of params to train
 #        CW_CP_V = T.tensordot(CW_CP, V, [[2], [0]])  # batch, seqlen, F, w*M+D
 #        CP_CW = T.concatenate([CP_stack, CW], 2)  # batch, seqlen, w*M+D
 #        CW_CP_V_re = CW_CP_V.reshape((curr_batch_size*img_h, F, concat_dim))  # batch*seqlen, F, w*M+D
 #        CP_CW_re = CP_CW.reshape((curr_batch_size*img_h, concat_dim))  # batch*seqlen, W*M+D
 #        result = T.batched_dot(CW_CP_V_re, CP_CW_re)  # batch*seqlen, F
 #        result = result.reshape((curr_batch_size, img_h, F))  # batch, seqlen, F
 #        layer0_input_first = result.reshape((curr_batch_size, 1, img_h, F))  # batch, 1, seqlen, F
 #
 #        # second term
 #        CP_stack_re = CP_stack.reshape((curr_batch_size, 1, img_h, Tags.shape[1]*window))  # batch, 1, seq, M*w
 #        CW_re = CW.reshape((curr_batch_size, 1, img_h, Words.shape[1]))
 #        CW_CP_re = T.concatenate([CW_re, CP_stack_re], 3)  # batch, 1, seqlen, w*M+D
 #        Q = theano.shared(np.asarray(rng.uniform(low=-0.01,
 #                                                 high=0.01,
 #                                                 size=[concat_dim, F]),
 #                                     dtype=theano.config.floatX),
 #                          borrow=True,
 #                          name="Q")
 #        emb_layer_params += [Q]  # add Q to the list of params to train
 #        layer0_input_second = T.dot(CW_CP_re, Q)  # curr_batch_size, 1, img_h, F
 #        layer0_input = ReLU(layer0_input_first + layer0_input_second)
 #        img_w = F
 #
 #    elif model == "tensor":
 #        print 'use mix...'
 #        # first term
 #        words = Words[T.cast(x.flatten(), dtype="int32")].reshape((curr_batch_size*img_h, Words.shape[1]))
 #        tags = Tags[T.cast(z.flatten(), dtype="int32")].reshape((curr_batch_size*img_h, Tags.shape[1]))
 #        words_tags = T.concatenate([words, tags], 1)  # batch * seqlen, D+M
 #        tags_words = T.concatenate([tags, words], 1)  # batch * seqlen, D+M
 #        concat_dim = W.shape[1] + P.shape[1]
 #        F = 150  # TODO: set final dim for each token
 #        V = theano.shared(np.asarray(rng.uniform(low=-0.01,
 #                                                 high=0.01,
 #                                                 size=[concat_dim, F, concat_dim]),  # D+M, F, D+M
 #                                     dtype=theano.config.floatX),
 #                          borrow=True,
 #                          name="V")
 #        emb_layer_params += [V]
 #        words_tags_V = T.tensordot(words_tags, V, [[1], [0]])  # batch * seqlen, F, D+M
 #        mix_vec = T.batched_dot(words_tags_V, tags_words)  # batch * seqlen, F
 #
 #        # second term
 #        Q = theano.shared(np.asarray(rng.uniform(low=-0.01,
 #                                                 high=0.01,
 #                                                 size=[concat_dim, F]),
 #                                     dtype=theano.config.floatX),
 #                          borrow=True,
 #                          name="Q")
 #        emb_layer_params += [Q]
 #        merge_vec = T.dot(words_tags, Q)  # batch * seqlen, F
 #        layer0_input = ReLU(mix_vec + merge_vec).reshape((curr_batch_size, 1, img_h, F))
 #        img_w = F