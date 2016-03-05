import numpy as np
import cPickle
from collections import defaultdict
import re
import pandas as pd
from nltk.tag import StanfordPOSTagger


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    pos_vocab = defaultdict(float)
    pos_tagger = StanfordPOSTagger(
        'pos-tag/english-left3words-distsim.tagger',
        'pos-tag/stanford-postagger.jar',
        'utf8', False, '-mx2000m')

    with open(pos_file, "rb") as f:
        revs_text = []
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            revs_text.append(orig_rev.split())

        revs_tagged = pos_tagger.tag_sents(revs_text)

        for rev_tagged in revs_tagged:
            text = list(zip(*rev_tagged)[0])
            tag = list(zip(*rev_tagged)[1])
            words = set(text)
            for word in words:
                vocab[word] += 1
            postags = set(tag)
            for postag in postags:
                pos_vocab[postag] += 1
            datum = {"y": 1,
                     "text": ' '.join(text),
                     "tag": ' '.join(tag),
                     "num_words": len(text),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    with open(neg_file, "rb") as f:
        revs_text = []
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            revs_text.append(orig_rev.split())

        revs_tagged = pos_tagger.tag_sents(revs_text)

        for rev_tagged in revs_tagged:
            text = list(zip(*rev_tagged)[0])
            tag = list(zip(*rev_tagged)[1])
            words = set(text)
            for word in words:
                vocab[word] += 1
            postags = set(tag)
            for postag in postags:
                pos_vocab[postag] += 1
            datum = {"y": 0,
                     "text": ' '.join(text),
                     "tag": ' '.join(tag),
                     "num_words": len(text),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    return revs, vocab, pos_vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads embeddings from bin file
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs, layer1_size


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


if __name__=="__main__":    
    w2v_file = "data/GoogleNews-vectors-negative300.bin"
    pos_emb_file = "data/1billion-pos-48.bin"
    data_folder = ["mr/rt-polarity.pos", "mr/rt-polarity.neg"]

    print "loading data...",
    revs, vocab, pos_vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)

    print "loading word embeddings...",
    w2v, w2v_dim = load_bin_vec(w2v_file, vocab)
    print "word embeddings loaded!"
    print "pretrained num words: " + str(len(w2v))
    add_unknown_words(w2v, vocab, k=w2v_dim)
    W, word_idx_map = get_W(w2v, k=w2v_dim)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k=w2v_dim)
    W2, _ = get_W(rand_vecs, k=w2v_dim)

    print "loading pos embeddings...",
    p2v, p2v_dim = load_bin_vec(pos_emb_file, pos_vocab)
    print "pos embeddings loaded!"
    print "pretrained num pos tags: " + str(len(p2v))
    add_unknown_words(p2v, pos_vocab, k=p2v_dim)
    P, pos_idx_map = get_W(p2v, k=p2v_dim)

    cPickle.dump([revs, W, W2, word_idx_map, vocab, P, pos_idx_map], open("mr.p", "wb"))
    print "dataset created!"
