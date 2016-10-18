from itertools import islice
from collections import Counter
import zipfile
import numpy as np
import tensorflow as tf

text_file = "/Users/dracz/data/text8.zip"


def read_zipped_text(filename=text_file):
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
    def __init__(self, text, n=2, max_vocab=None):
        """
        Initialize the encoder
        :param text: Text for initialization
        :param n: Size of the gram
        :param max_vocab: Maximum size of vocabulary
        """
        d = Counter(ch_grams(text, n=n, overlap=n-1))
        self.n = n
        self.grams = [s for s, cnt in d.most_common(max_vocab)]
        self.vocab_size = len(self.grams) + 1
        self.unk_value = 0
        self._enc_dict = {c: i+1 for i, c in enumerate(self.grams)}
        self._dec_dict = {i+1: c for i, c in enumerate(self.grams)}
        del d

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

    def batches(self, text, seq_len=2, batch_size=2, max_batches=None):
        """
        Generate batches of encoded n-gram sequences
        :param text: The text to extract batches from
        :param seq_len: The number of grams per example
        :param batch_size: The number of example sequences per batch
        :param max_batches: Max number of batches, or unlimited if None
        :return: A generator over batches of one-hot encoded n-gram sequences
        """
        for s in text_batches(text, self.n, seq_len, batch_size, max_batches):
            yield self.batch(s, seq_len, batch_size)

    def batch(self, text, seq_len, batch_size):
        """
        Generate a single batch from the text of shape
        :param text: Text to extract batch from
        :param seq_len: The length of the n-gram sequences
        :param batch_size: The number of sequences in each batch
        :return: A batch of one-hot encoded n-gram sequences of shape (batch_size, seq_len, vocab_size)
        """
        assert len(text) == self.n * seq_len * batch_size
        a = [self.encode_onehot(g) for g in ch_grams(text, n=self.n)]
        return np.asarray(a).reshape((batch_size, seq_len, self.vocab_size))


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


def _print_header(msg, hr="-" * 80):
    """Print a header for testing and debugging"""
    print("{}\n{}\n{}".format(hr, msg, hr))


def _test_encoder():
    """Test the character gram encoder/decoder"""
    _print_header("Testing character gram encoding/decoding...")
    text = read_zipped_text(text_file)[:24]
    for n in range(1, 5):
        _print_header("n = {}".format(n))
        enc = GramEncoder(text, n=n)
        s = text
        ce = [enc.encode_gram(g) for g in ch_grams(s, n=n, overlap=0)]
        cd = [enc.decode_gram(i) for i in ce]
        cds = ''.join(cd)
        print('rawtext: "{}"'.format(s))
        print('encoded:', ce)
        print('decoded:', cd)
        print('decoded: "{}"'.format(cds))
        assert cds == s


def _test_text_batches(text_size=24, n=1, seq_len=4, batch_size=4):
    """Test the text batches"""
    text = read_zipped_text(text_file)[:text_size]
    _print_header("Testing text batches...")
    bb = list(text_batches(text, n=n, seq_len=seq_len, batch_size=batch_size))
    print("'{}' --> {}".format(text, bb))
    assert(len(bb) == text_size - n * seq_len * batch_size + 1)


def _test_batches(text_size=34, n=2, seq_len=4, batch_size=4):
    """Test the generated batches of onehot-encoded gram sequences"""
    text = read_zipped_text(text_file)[:text_size]
    enc = GramEncoder(text, n)
    _print_header("Testing batches...")
    print("text: '{}'".format(text))
    print("batch_size:", batch_size, "seq_len:", seq_len)
    bb = list(enc.batches(text, seq_len, batch_size))
    for i, b in enumerate(bb):
        print("\nBatch:", i + 1)
        for seq in b:
            d = [enc.decode_onehot(v) for v in seq]
            print("{}".format(d))


if __name__ == "__main__":
    _test_encoder()
    _test_text_batches()
    _test_batches()

