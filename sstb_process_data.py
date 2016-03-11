import numpy as np
import cPickle
from collections import defaultdict
import re
import pandas as pd
from nltk.tag import StanfordPOSTagger
import csv
import sys


def get_split_num(split):
    if split == 'train' or split == 'train_phrases':
        return 0
    elif split == 'test':
        return 1
    elif split == 'dev':
        return 2
    return -1


def build_data_cv(data_file, all_phrases, min_len=3):
    revs = []
    vocab = defaultdict(float)
    pos_vocab = defaultdict(float)
    pos_tagger = StanfordPOSTagger(
        'pos-tag/english-left3words-distsim.tagger',
        'pos-tag/stanford-postagger.jar',
        'utf8', False, '-mx2000m')
    splits = ['train', 'test', 'dev']

    for split in splits:
        if split == 'train' and all_phrases:
            split = 'train_phrases'
        with open(data_file.format(split), "rb") as f:
            reader = csv.reader(f)
            revs_text = []
            sents = []
            for row in reader:
                rev, sent = row[0], int(row[1])
                rev = clean_str_sst(rev)
                rev_tokens = rev.split()
                if split == 'train_phrases' and len(rev_tokens) < min_len:
                    continue
                revs_text.append(rev_tokens)
                sents.append(sent)
            revs_tagged = pos_tagger.tag_sents(revs_text)
            for i in range(len(revs_tagged)):
                rev_tagged = revs_tagged[i]
                text = list(zip(*rev_tagged)[0])
                tag = list(zip(*rev_tagged)[1])
                for word in set(text):
                    vocab[word] += 1
                for postag in set(tag):
                    pos_vocab[postag] += 1
                rev_datum = {"y": sents[i],
                             "text": ' '.join(text),
                             "tag": ' '.join(tag),
                             "num_words": len(text),
                             "split": get_split_num(split)}
                revs.append(rev_datum)

    return revs, vocab, pos_vocab


def get_W(word_vecs, k):
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
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    w2v_file = "data/GoogleNews-vectors-negative300.bin"
    pos_emb_file = "data/1billion-pos-24.bin"
    data_file = "sstb/sstb_condensed_{}.csv"

    if len(sys.argv) < 2 or sys.argv[1] == 'reviews':
        all_phrases = False
    elif sys.argv[1] == 'phrases':
        all_phrases = True
    else:
        print 'invalid argument'
        sys.exit()

    print "loading sstb data...",
    revs, vocab, pos_vocab = build_data_cv(data_file, all_phrases)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "pos vocab size: " + str(len(pos_vocab))
    print "max sentence length: " + str(max_l)

    print "loading word embeddings...",
    w2v, w2v_dim = load_bin_vec(w2v_file, vocab)
    print "word embeddings loaded!"
    print "pretrained num words: " + str(len(w2v))
    add_unknown_words(w2v, vocab, k=w2v_dim)
    W, word_idx_map = get_W(w2v, k=w2v_dim)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k=w2v_dim)
    W_rand, _ = get_W(rand_vecs, k=w2v_dim)

    print "loading pos embeddings...",
    p2v, p2v_dim = load_bin_vec(pos_emb_file, pos_vocab)
    print "pos embeddings loaded!"
    print "pretrained num pos tags: " + str(len(p2v))
    add_unknown_words(p2v, pos_vocab, k=p2v_dim)
    P, pos_idx_map = get_W(p2v, k=p2v_dim)

    rand_vecs = {}
    add_unknown_words(rand_vecs, pos_vocab, k=p2v_dim)
    P_rand, _ = get_W(rand_vecs, k=p2v_dim)

    cPickle.dump([revs, W, W_rand, word_idx_map, vocab, P, P_rand, pos_idx_map, 1, 5], open("sstb.p", "wb"))
    print "dataset created!"
