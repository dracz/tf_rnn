import os
import math
import numpy as np
import tensorflow as tf
from data_io import GramEncoder, maybe_download_text8, read_zipped_text
from matplotlib import pyplot as plt


class Gram2Vec():

    def __init__(self, opts, sess):
        self._opts = opts
        self._sess = sess
        self._enc = GramEncoder.load(opts['encoder_file'])
        self.build_graph()
        tf.initialize_all_variables().run()
        if os.path.exists(opts['model_file']):
            saver = tf.train.Saver()
            saver.restore(sess, opts['model_file'])
            print("Model restored from '{}'.".format(opts['model_file']))


    def build_graph(self):
        print("Building graph...")
        opts = self._opts

        # Placeholders for examples, labels, validation
        self.x = tf.placeholder(tf.int32, shape=[opts['batch_size']])
        self.y = tf.placeholder(tf.int32, shape=[opts['batch_size'], 1])

        # Embedding weights
        init_r = 0.5 / opts['emb_dim']
        w_emb = tf.Variable(tf.random_uniform([self._enc.vocab_size, opts['emb_dim']], -init_r, init_r))

        # Output weights (transposed) and biases
        init_std = 1.0 / math.sqrt(opts['emb_dim'])
        w = tf.Variable(tf.truncated_normal([self._enc.vocab_size, opts['emb_dim']], stddev=init_std))
        b = tf.Variable(tf.zeros([self._enc.vocab_size]))

        # Embedding lookup and loss computation
        embed = tf.nn.embedding_lookup(w_emb, self.x)
        nce_loss = tf.nn.nce_loss(w, b, embed, self.y, opts['neg_samples'], self._enc.vocab_size)
        self.loss = tf.reduce_mean(nce_loss)

        # Optimization
        self.optimizer = tf.train.GradientDescentOptimizer(opts['lr']).minimize(self.loss)

        # Validation and test graphs
        x_sample = np.random.choice(opts['valid_head'], opts['valid_size'], replace=False) + 1
        self.x_valid = tf.constant(x_sample, dtype=tf.int32)
        self.x_test = tf.placeholder(tf.int32)
        self.norm_embed = w_emb / tf.sqrt(tf.reduce_sum(tf.square(w_emb), 1, keep_dims=True))

        test_embed = tf.nn.embedding_lookup(self.norm_embed, self.x_test)
        valid_embed = tf.nn.embedding_lookup(self.norm_embed, self.x_valid)
        self.similarity = tf.matmul(valid_embed, self.norm_embed, transpose_b=True)
        self.similarity_test = tf.matmul(test_embed, self.norm_embed, transpose_b=True)
        self._saver = tf.train.Saver()

    def similar(self, i, k):
        """Get the top-k most similar embeddings to specified gram id """
        sim = self._sess.run(self.similarity_test, feed_dict={self.x_test: [i]})
        return (-sim).argsort()[0, 1:k+1]

    def train(self, text):
        print("Beginning training procedure...")
        opts = self._opts
        batches = self._enc.skipgram_batches(text,
                                             window_size=opts["window_size"],
                                             batch_size=opts["batch_size"],
                                             max_batches=opts["epochs"])

        # iterate through batches and keep track of average loss
        total_loss, steps = 0, 0
        last_loss = None
        losses = []

        for step, batch in enumerate(batches):
            feed_dict = {self.x: batch[0], self.y: batch[1]}
            _, step_loss = self._sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            if step % 100 == 0:
                losses.append((step, step_loss))
            total_loss += step_loss
            steps += 1
            if step > 0 and step % opts["log_every"] == 0:
                avg_loss = total_loss / steps
                pct = 0 if last_loss is None else (avg_loss - last_loss)*100. / avg_loss
                print("Step: {:<7} Loss/batch: {:<7.3f} ({:.3f}%)".format(step, avg_loss, pct))
                last_loss = avg_loss
                total_loss, steps = 0, 0

            if step % (opts["log_every"] * 5) == 0:
                sim = self.similarity.eval()
                for i in range(opts["valid_size"]):
                    valid_gram = self._enc.decode_gram(self.x_valid[i].eval())
                    top_k = 5
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    w = [self._enc.decode_gram(nearest[k]) for k in range(top_k)]
                    print("Nearest to '{}': {}".format(valid_gram, w))

        self.final_embed = self.norm_embed.eval()

        losses.append((step, step_loss))
        save_path = self._saver.save(self._sess, opts["model_file"])

        print("Model saved to: '{}'".format(save_path))
        plt.plot([x for x, y in losses], [y for x, y in losses])
        plt.show()


def train(text, opts):
    print("Options:", opts)
    with tf.Graph().as_default(), tf.Session() as session:
        model = Gram2Vec(opts, session)
        model.train(text)


def maybe_encode_bigrams(text, filename):
    if os.path.exists(filename):
        print("Found {}".format(filename))
        return GramEncoder.load(filename)
    else:
        print("Encoding...")
        enc = GramEncoder.load_text(text, 2)
        enc.save(filename)
        return enc


def fmt_dict(d):
    return '\n'.join(["'{}': {}".format(k, v) for k,v in d.items()])


opts = dict(n=2, batch_size=1024, emb_dim=128, window_size=1, neg_samples=64,
            lr=0.5, epochs=10000, valid_size=8, valid_head=100, log_every=100,
            model_file="bigram_emb.ckpt", encoder_file="text8_bigrams.pkl")


def train_text8(opts):
    text8_zip = maybe_download_text8()
    text = read_zipped_text(text8_zip)
    maybe_encode_bigrams(text, opts["encoder_file"]).save(opts["encoder_file"])
    train(text, opts)


def test_text8(opts, k=5, n=100):
    with tf.Graph().as_default(), tf.Session() as session:
        g2v = Gram2Vec(opts, session)
        enc = GramEncoder.load(opts['encoder_file'])
        for i in range(0, n+1):
            g = enc.decode_gram(i)
            v = g2v.similar(i, k)
            n = [enc.decode_gram(j) for j in v]
            print("'{}' --> {}".format(g, n))

if __name__ == "__main__":
    train_text8(opts)
    test_text8(opts)
