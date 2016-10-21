from itertools import islice
from collections import Counter
from time import clock
import urllib.request
import os
import pickle
import zipfile

import numpy as np
import tensorflow as tf

text8_zip = 'text8.zip'


def maybe_download_text8():
    return maybe_download('http://mattmahoney.net/dc/text8.zip', text8_zip, 31344016)


def maybe_download(url, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("Downloading {} --> {}...".format(url, filename))
        filename, _ = urllib.request.urlretrieve(url, filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_zipped_text(filename=text8_zip):
    """Read zipped text file as text string"""
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


def n_grams(s, n=2, overlap=0):
    """
    Split a seq into tuples of grams
    :param s: Sequence to split
    :param n: Length of the n-gram
    :param overlap: Degree of overlap between adjacent grams
    :return: A sequence of tuples, each containing n elements from s
    """
    assert n > 0 and 0 <= overlap < n
    return zip(*[islice(s, i, None, n-overlap) for i in range(n)])


def ch_grams(text, n=2, overlap=0):
    """
    Split a string into character n-grams
    :param text: String to split
    :param n: Number of characters in each gram
    :param overlap: Degree of overlap between adjacent grams
    :return: A sequence of character n-grams as strings
    """
    return [''.join(t) for t in n_grams(text, n, overlap)]


class GramEncoder:
    """An encoder/decoder for character n-grams"""

    @staticmethod
    def load(filename):
        """
        Load a saved encoder from pickle file
        :param filename: path to file containing previously saved encoder
        """
        with open(filename, 'rb') as f:
            n, grams = pickle.load(f)
        return GramEncoder(n, grams)

    @staticmethod
    def load_text(text, n=2, max_vocab=None, save_file=None):
        """
        Initialize an character gram encoder/decoder from text
        :param text: Text for initialization
        :param n: Size of the gram
        :param max_vocab: Maximum size of vocabulary
        :param save_file: If not None, then save encoder to file
        """
        t = clock()
        d = Counter(ch_grams(text, n=n, overlap=n-1))
        grams = [s for s, cnt in d.most_common(max_vocab)]
        del d
        print("Loaded {:,} characters in {:,.2f} secs".format(len(text), clock() - t))
        enc = GramEncoder(n, grams)
        if save_file is not None:
            enc.save(save_file)
        return enc

    def __init__(self, n, grams):
        """
        Init character n-gram encoder
        :param n: Gram length
        :param grams: List of grams
        """
        self.n = n
        self.grams = grams
        self.vocab_size = len(grams) + 1
        self.unk_value = 0
        self._enc_dict = {g: i+1 for i, g in enumerate(grams)}
        self._dec_dict = {i+1: g for i, g in enumerate(grams)}

    def save(self, filename):
        """
        Save encoder to file
        :param filename: File to save to
        """
        print("Saving {} to '{}'...".format(self.__class__.__name__, filename))
        with open(filename, 'wb') as f:
            pickle.dump((self.n, self.grams), f, pickle.HIGHEST_PROTOCOL)

    def encode_gram(self, s):
        """
        Encode a string as an n-gram
        :param s: A string of length n to encode
        :return: Int id of the encoded n-gram
        """
        if len(s) != self.n:
            raise ValueError('length of string to encode must be == n')
        return self._enc_dict.get(s, self.unk_value)

    def encode_onehot(self, s):
        """
        Encode the string as a one-hot encoded n-gram
        :param s: String to encode as one-hot n-gram
        :return: A float vector of length vocab_size
        """
        v = np.zeros(shape=(self.vocab_size, 1), dtype=np.float)
        v[self.encode_gram(s)] = 1.0
        return v

    def decode_gram(self, i, default='_'):
        """
        Decode gram id into string, or default value if id is not found
        :param i: The int n-gram id
        :param default: Default char to return if id not found
        :return: The decoded gram as string
        """
        return self._dec_dict.get(i, default * self.n)

    def decode_onehot(self, v):
        """
        Decode one-hot or probability vector to most likely gram
        :param v: One-hot/probability encoded float vector
        :return: The decoded n-gram as string
        """
        return self.decode_gram(np.argmax(v))

    def gram_batches(self, text, seq_len=5, batch_size=32, max_batches=None):
        """
        Generate batches of encoded n-gram sequences
        :param text: The text to extract batches from
        :param seq_len: The number of grams per example
        :param batch_size: The number of example sequences per batch
        :param max_batches: Max number of batches, or unlimited if None
        :return: A generator over batches of one-hot encoded n-gram sequences
        """
        for s in text_batches(text, self.n, seq_len, batch_size, max_batches):
            a = np.asarray([self.encode_gram(g) for g in ch_grams(s, n=self.n)])
            yield a.reshape((batch_size, seq_len))

    def onehot_batches(self, text, seq_len=5, batch_size=32, max_batches=None):
        """
        Generate batches of encoded n-gram sequences
        :param text: The text to extract batches from
        :param seq_len: The number of grams per example
        :param batch_size: The number of example sequences per batch
        :param max_batches: Max number of batches, or unlimited if None
        :return: A generator over batches of one-hot encoded n-gram sequences
        """
        for s in text_batches(text, self.n, seq_len, batch_size, max_batches):
            a = [self.encode_onehot(g) for g in ch_grams(s, n=self.n)]
            yield np.asarray(a).reshape((batch_size, seq_len, self.vocab_size))

    def skipgram_batches(self, text, window_size=4, batch_size=32, max_batches=None):
        """
        Generate batches for training skip-gram model: http://arxiv.org/pdf/1301.3781.pdf
        :param text: The text to extract batches from
        :param window_size: Number of grams on each side of the target word to include in context
        :return: Batch of (target, context) pairs
        """
        assert batch_size % (2 * window_size) == 0
        sz = batch_size // (2 * window_size)  # how many gram batches needed
        for b in self.gram_batches(text, window_size * 2 + 1, sz, max_batches):
            x, y = [], []
            ctx = list(range(window_size)) + list(range(window_size + 1, window_size * 2 + 1))
            for seq in b:
                for i in ctx:
                    x.append(seq[window_size])
                    y.append(seq[i])
            yield np.asarray(x), np.asarray(y).reshape((batch_size, 1))


def text_batches(text, n=2, seq_len=2, batch_size=2, max_batches=None):
    """
    Generate batches of character-gram sequences as text
    :param text: The text to extract from
    :param n: The number of characters per gram
    :param seq_len: The number of grams per example
    :param batch_size: The number of example sequences per batch
    :param max_batches: Max number of batches, or unlimited if None
    :return: Generator over batches of char-gram sequences
    """
    num_batches = 0
    for start_pos in range(seq_len * n * batch_size):
        batch = n * seq_len * batch_size
        for i in range(start_pos, len(text) - batch + 1, batch):
            yield text[i:i + batch_size * seq_len * n]
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                return


# Test code below

def _print_header(msg, hr="#"):
    """Print a header for testing and debugging"""
    print("\n{}\n{} {}\n{}".format(hr * 80, hr, msg, hr * 80))


def _read_test_text(text_size=100):
    """read test text"""
    text = read_zipped_text(text8_zip)[:text_size]
    print("Text: '{}'".format(text))
    return text


def _test_encoder():
    """Test the character gram encoder/decoder"""
    _print_header("Testing character gram encoding/decoding...")
    text = _read_test_text()
    for n in range(1, 5):
        _print_header("n = {}".format(n))
        enc = GramEncoder.load_text(text, n)
        s = text
        ce = [enc.encode_gram(g) for g in ch_grams(s, n=n, overlap=0)]
        cd = [enc.decode_gram(i) for i in ce]
        cds = ''.join(cd)
        print('encoded:', ce)
        print('decoded:', cd)
        print('decoded: "{}"'.format(cds))


def _test_text_batches(text_size=24, n=1, seq_len=4, batch_size=4):
    """Test the text batches"""
    _print_header("Testing text batches... {}".format(locals()))
    text = _read_test_text(text_size)
    bb = list(text_batches(text, n=n, seq_len=seq_len, batch_size=batch_size))
    for b in bb:
        print(b)
    assert(len(bb) == text_size - n * seq_len * batch_size + 1)


def _test_gram_batches(text_size=48, n=2, seq_len=4, batch_size=4):
    """Test batches of gram ids"""
    _print_header("Testing gram batches... {}".format(locals()))
    text = _read_test_text(text_size)
    enc = GramEncoder.load_text(text, n)
    bb = list(enc.gram_batches(text, seq_len, batch_size))
    for i, b in enumerate(bb):
        print("\nBatch {}: {} \n{}\n -->".format(i + 1, b.shape, b))
        for seq in b:
            d = [enc.decode_gram(v) for v in seq]
            print("{}".format(d))
        assert b.shape == (batch_size, seq_len)


def _test_onehot_batches(text_size=34, n=2, seq_len=4, batch_size=4):
    """Test the generated batches of onehot-encoded gram sequences"""
    _print_header("Testing batches... {}".format(locals()))
    text = _read_test_text(text_size)
    enc = GramEncoder.load_text(text, n)
    bb = list(enc.onehot_batches(text, seq_len, batch_size))
    for i, b in enumerate(bb):
        print("\nBatch {}: {}\n{}\n -->".format(i + 1, b.shape, b))
        for seq in b:
            d = [enc.decode_onehot(v) for v in seq]
            print("{}".format(d))
        assert b.shape == (batch_size, seq_len, enc.vocab_size)


def _test_skipgram_batches(text_size=500, n=2, window_size=3, batch_size=12):
    """Test the skip-gram batches"""
    _print_header("Testing skip-gram batches... {}".format(locals()))
    text = _read_test_text(text_size)
    enc = GramEncoder.load_text(text, n)
    batches = enc.skipgram_batches(text, window_size, batch_size)
    for i, batch in enumerate(batches):
        x, y = batch
        assert len(x) == len(y) == batch_size
        print("\nBatch {}: {}, {}\n -->".format(i + 1, x.shape, y.shape))
        print(x)
        print(y)
        for t in zip(x, y):
            print(enc.decode_gram(t[0]), enc.decode_gram(t[1]))



def _test_save_load():
    fname = "_test_bigrams.pkl"
    text8 = read_zipped_text()[:10000]
    enc1 = GramEncoder.load_text(text8, 2)
    enc1.save(fname)
    enc2 = GramEncoder.load(fname)
    assert enc1.vocab_size == enc2.vocab_size
    assert enc1.grams == enc2.grams
    for i in range(enc1.vocab_size):
        assert enc1.decode_gram(i) == enc2.decode_gram(i)
    for g in enc1.grams:
        assert enc1.encode_gram(g) == enc2.encode_gram(g)
    os.remove(fname)


if __name__ == "__main__":
    '''
    maybe_download_text8()
    _test_encoder()
    _test_text_batches()
    _test_gram_batches()
    _test_onehot_batches()
    _test_save_load()
    '''
    _test_skipgram_batches()




