"""
Vocabulary module for conversion between word tokens and numerical indices.
"""
import warnings
from collections import Counter, namedtuple
import numpy as np
import json
import zipfile
from copy import deepcopy
import logging
import utils


__author__ = 'victor'


class OutOfVocabularyWordException(Exception):
    pass


class SimpleVocab(object):
    """A mapping between words/labels and numerical indices, without unknown words/labels.
    
    Example:

    .. code-block:: python

        v = SimpleVocab()
        indices = v.add("I'm a list of words".split())
        print('indices')

    NOTE: UNK is always represented by the 0 index.
    """

    def __init__(self):
        """Construct a SimpleVocab object.
        """

        self._counts = Counter()

        self._index2word = []
        self._word2index = {}

    def __getitem__(self, word):
        """Get the index for a word.

        If the word is unknown, a KeyError is raised.
        """
        if word not in self._word2index:
            raise OutOfVocabularyWordException('Word "{}" is not in the vocabulary'.format(word))
        return self._word2index[word]

    def __contains__(self, word):
        return word in self._word2index

    def __len__(self):
        return len(self._index2word)

    def __str__(self):
        return 'Vocab(%d words)' % len(self)

    def __iter__(self):
        for w in self._index2word:
            yield w

    def copy(self, keep_words=True):
        if not keep_words:
            return self.__class__()
        else:
            return deepcopy(self)

    def add(self, word, count=1):
        """Add a word to the vocabulary and return its index.

        :param word: word or a list of words to add to the dictionary.

        :param count: how many times to add the word.

        :return: index/indices of the added word(s).

        WARNING: this function assumes that if the Vocab currently has N words, then
        there is a perfect bijection between these N words and the integers 0 through N-1.
        """
        if isinstance(word, list):
            return [self.add(w) for w in word]

        if word not in self:
            self._word2index[word] = len(self)
            self._index2word.append(word)
        self._counts[word] += count
        return self[word]

    def word2index(self, word):
        return [self.word2index(w) for w in word] if isinstance(word, list) else self[word]

    def words2indices(self, words): return self.word2index(words)

    def index2word(self, index):
        return [self.index2word(i) for i in index] if isinstance(index, list) else  self._index2word[index]

    def indices2words(self, indices): return self.index2word(indices)

    def count(self, w):
        return self._counts[w]

    def union(self, another):
        """
        The ordering of the indices will have self._words2index first, then another._words2index
        """
        v = self.copy(keep_words=True)
        for w, c in another._counts.items():
            v.add(w, count=c)
        return v

    def intersection(self, another):
        words = set(self._index2word).intersection(set(another._index2word))
        v = self.copy(keep_words=False)
        for w in words:
            v.add(w, self.count(w) + another.count(w))
        return v

    def difference(self, another):
        words = set(self._index2word).difference(another)
        v = self.copy(keep_words=False)
        for w in words:
            v.add(w, self.count(w) + another.count(w))
        return v

    def serialize(self, fname):
        """Write vocab to a file.
        """
        with open(fname, 'w') as f:
            json.dump({
                'counts': dict(self._counts),
                'index2word': self._index2word,
            }, f)

    @classmethod
    def deserialize(cls, fname):
        """Load vocab from a file.
        """
        with open(fname) as f:
            d = json.load(f)
            v = cls()
            v._counts = Counter(d['counts'])
            v._index2word = d['index2word']
            v._word2index = {w: i for i, w in enumerate(d['index2word'])}
        return v


class Vocab(SimpleVocab):
    """A mapping between words and numerical indices. This class is used to facilitate the creation of word embedding matrices.

    Example:

    .. code-block:: python

        v = Vocab()
        indices = v.add("I'm a list of words".split())
        print('indices')

    NOTE: UNK is always represented by the 0 index.
    NOTE: PAD is always represented by the 1 index.
    """

    def __init__(self, unk='<UNK>', pad='<PAD>'):
        """Construct a Vocab object.

        :param unk: string to represent the unknown word (UNK). It is always represented by the 0 index.
        :param pad: string to represent the unknown word (PAD). It is always represented by the 1 index.
        """
        super(Vocab, self).__init__()
        self.UNK_INDEX = 0
        self.unk = unk
        self.PAD_INDEX = 1
        self.pad = pad

        # assign an index for UNK
        self.add(self.unk, count=0)
        self.add(self.pad, count=0)

    def __getitem__(self, word):
        """Get the index for a word.

        If the word is unknown, the index for UNK is returned.
        """
        return self._word2index.get(word, self.UNK_INDEX)

    def copy(self, keep_words=True):
        if not keep_words:
            return self.__class__(self.unk, self.pad)
        else:
            return deepcopy(self)

    def prune_rares(self, cutoff=2):
        warnings.warn("Use prune_rares_by_cutoff instead.", DeprecationWarning)
        return self.prune_rares_by_cutoff(cutoff)

    def prune_rares_by_cutoff(self, cutoff=2):
        """
        returns a **new** `Vocab` object that is similar to this one but with rare words removed. Note that the indices in the new `Vocab` will be remapped (because rare words will have been removed).

        :param cutoff: words occuring less than this number of times are removed from the vocabulary.

        :return: A new, pruned, vocabulary.

        NOTE: UNK is never pruned.
        """
        # make a deep copy and reset its contents
        v = self.copy(keep_words=False)
        for w in self:
            if self._counts[w] >= cutoff or w == self.unk or w == self.pad:  # don't remove unk or pad
                v.add(w, count=self._counts[w])
        return v

    def prune_rares_by_top_k(self, k):
        """
        k = desired vocab size
        """
        sorted_by_count = self._counts.most_common(min(len(self), k))
        v = self.copy(keep_words=False)
        for w, c in sorted_by_count:
            v.add(w, c)
        return v

    def serialize(self, fname):
        """Write vocab to a file.
        """
        with open(fname, 'w') as f:
            json.dump({
                'counts': dict(self._counts),
                'unk': self.unk,
                'pad': self.pad,
                'index2word': self._index2word,
            }, f)

    @classmethod
    def deserialize(cls, fname):
        """Load vocab from a file.
        """
        with open(fname) as f:
            d = json.load(f)
            v = cls(unk=d['unk'], pad=d['pad'])
            v._counts = Counter(d['counts'])
            v._index2word = d['index2word']
            v._word2index = {w: i for i, w in enumerate(d['index2word'])}
        return v


class EmbeddedVocab(Vocab):

    def __init__(self, unk='<UNK>', pad='<PAD>'):
        super(EmbeddedVocab, self).__init__(unk=unk, pad=pad)
        self.E = None

    def get_embeddings(self):
        """
        :return: the embedding matrix for this vocabulary object.
        """
        raise NotImplementedError()

    def load_embeddings(self, **kwargs):
        if self.E is None:
            logging.info('Retrieving Embeddings')
            self.E = self.get_embeddings(**kwargs)
        return self.E

    def serialize(self, fname):
        """Write vocab to a file.
        """
        super(EmbeddedVocab, self).serialize(fname)
        np.save(fname + '.npy', self.E)
        #np.save(fname.split('.')[0] + '.npy', self.E)

    @classmethod
    def deserialize(cls, fname):
        """Load vocab from a file.
        """
        v = super(EmbeddedVocab, cls).deserialize(fname)
        v.E = np.load(fname + '.npy')
        #v.E = np.load(fname.split('.')[0] + '.npy')
        return v

    def _backfill_unk_emb(self, E, filled_words):
        """ Backfills an embedding matrix with the embedding for the unknown token.

        :param E: original embedding matrix of dimensions `(vocab_size, emb_dim)`.
        :param filled_words: these words will not be backfilled with unk.

        NOTE: this function is for internal use.
        """
        unk_emb = E[self[self.unk]]
        for i, word in enumerate(self):
            if word not in filled_words:
                E[i] = unk_emb


class SennaVocab(EmbeddedVocab):

    """
    Vocab object with initialization from Senna by Collobert et al.

    Reference: http://ronan.collobert.com/senna
    """

    embeddings_url = 'https://github.com/baojie/senna/raw/master/embeddings/embeddings.txt'
    words_url = 'https://raw.githubusercontent.com/baojie/senna/master/hash/words.lst'
    n_dim = 50

    def __init__(self, unk='UNKNOWN', pad='PAD'):
        super(SennaVocab, self).__init__(unk=unk, pad=pad)

    @classmethod
    def gen_word_list(cls, fname):
        with open(fname) as f:
            for line in f:
                yield line.rstrip("\n\r")

    @classmethod
    def gen_embeddings(cls, fname):
        with open(fname) as f:
            for line in f:
                yield np.fromstring(line, sep=' ')

    def get_embeddings(self, rand=None, dtype='float32'):
        """
        Retrieves the embeddings for the vocabulary.

        :param rand: Random initialization function for out-of-vocabulary words. Defaults to `np.random.uniform(-0.1, 0.1, size=shape)`.
        :param dtype: Type of the matrix.
        :return: embeddings corresponding to the vocab instance.

        NOTE: this function will download potentially very large binary dumps the first time it is called.
        """
        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        embeddings = utils.get_data_or_download('senna', 'embeddings.txt', self.embeddings_url)
        words = utils.get_data_or_download('senna', 'words.lst', self.words_url)

        E = rand((len(self), self.n_dim)).astype(dtype)

        seen = []
        for word_emb in zip(self.gen_word_list(words), self.gen_embeddings(embeddings)):
            w, e = word_emb
            if w in self:
                seen += [w]
                E[self[w]] = e
        self._backfill_unk_emb(E, set(seen))
        return E



class GloveVocab(EmbeddedVocab):

    """
    Vocab object with initialization from GloVe by Pennington et al.

    Reference: http://nlp.stanford.edu/projects/glove
    """

    GloveSetting = namedtuple('GloveSetting', ['url', 'n_dims', 'size', 'description'])
    settings = {
        'common_crawl_48': GloveSetting('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                        [300], '1.75GB', '48B token common crawl'),
        'common_crawl_840': GloveSetting('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                         [300], '2.03GB', '840B token common crawl'),
        'twitter': GloveSetting('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                [25, 50, 100, 200], '1.42GB', '27B token twitter'),
        'wikipedia_gigaword': GloveSetting('http://nlp.stanford.edu/data/glove.6B.zip',
                                           [50, 100, 200, 300], '822MB', '6B token wikipedia 2014 + gigaword 5'),
    }

    def __init__(self, unk='<UNK>', pad='<PAD>'):
        super(GloveVocab, self).__init__(unk=unk, pad=pad)

    def get_embeddings(self, rand=None, dtype='float32', corpus='common_crawl_840', n_dim=300):
        """
        Retrieves the embeddings for the vocabulary.

        :param rand: Random initialization function for out-of-vocabulary words. Defaults to `np.random.uniform(-0.1, 0.1, size=shape)`.
        :param dtype: Type of the matrix.
        :param corpus: Corpus to use. Please see `GloveVocab.settings` for available corpus.
        :param n_dim: dimension of vectors to use. Please see `GloveVocab.settings` for available corpus.
        :return: embeddings corresponding to the vocab instance.

        NOTE: this function will download potentially very large binary dumps the first time it is called.
        """
        assert corpus in self.settings, '{} not in supported corpus {}'.format(corpus, self.settings.keys())
        self.n_dim, self.corpus, self.setting = n_dim, corpus, self.settings[corpus]
        assert n_dim in self.setting.n_dims, '{} not in supported dimensions {}'.format(n_dim, self.setting.n_dims)

        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        zip_file = utils.get_data_or_download('glove', '{}.zip'.format(self.corpus), self.setting.url, size=self.setting.size)

        E = rand((len(self), self.n_dim)).astype(dtype)
        n_dim = str(self.n_dim)

        with zipfile.ZipFile(zip_file) as zf:
            # should be only 1 txt file
            names = [info.filename for info in zf.infolist() if info.filename.endswith('.txt') and n_dim in info.filename]
            if not names:
                s = 'no .txt files found in zip file that matches {}-dim!'.format(n_dim)
                s += '\n available files: {}'.format(names)
                raise IOError(s)
            name = names[0]
            seen = []
            with zf.open(name) as f:
                for line in f:
                    toks = line.decode('utf-8').rstrip().split(' ')
                    word = toks[0]
                    if word in self:
                        seen += [word]
                        E[self[word]] = np.array([float(w) for w in toks[1:]], dtype=dtype)
            logging.info('{} embeddings loaded'.format(len(set(seen))))
            self._backfill_unk_emb(E, set(seen))
            return E
