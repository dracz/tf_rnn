import os
import math
import numpy as np
import tensorflow as tf
import data_io


class Gram2Vec():

    def __init__(self, text, opts, session):
        print("Initializing {}...".format(self.__class__.__name__))

        self._opts = opts
        self._session = session

        if opts["encoder_file"] is None:
            self._enc = data_io.GramEncoder.load_text(text, opts['n'])
        else:
            self._enc = data_io.GramEncoder.load(opts["encoder_file"])

        print("Building graph...")

        # Placeholders for examples, labels, validation
        x = tf.placeholder(tf.int32, shape=[opts['batch_size']])
        y = tf.placeholder(tf.int32, shape=[opts['batch_size'], 1])

        x_sample = np.random.choice(opts['valid_head'], opts['valid_size'], replace=False) + 1
        x_valid = tf.constant(x_sample, dtype=tf.int32)

        # Embedding weights
        init_r = 0.5 / opts['emb_dim']
        w_emb = tf.Variable(tf.random_uniform([self._enc.vocab_size, opts['emb_dim']], -init_r, init_r))

        # Output weights (transposed) and biases
        init_std = 1.0 / math.sqrt(opts['emb_dim'])
        w = tf.Variable(tf.truncated_normal([self._enc.vocab_size, opts['emb_dim']], stddev=init_std))
        b = tf.Variable(tf.zeros([self._enc.vocab_size]))

        # Embedding lookup and loss computation
        embed = tf.nn.embedding_lookup(w_emb, x)
        nce_loss = tf.nn.nce_loss(w, b, embed, y, opts['neg_samples'], self._enc.vocab_size)
        loss = tf.reduce_mean(nce_loss)

        # Optimization
        optimizer = tf.train.GradientDescentOptimizer(opts['lr']).minimize(loss)

        # Validation graph
        norm_embed = w_emb / tf.sqrt(tf.reduce_sum(tf.square(w_emb), 1, keep_dims=True))
        valid_embed = tf.nn.embedding_lookup(norm_embed, x_valid)
        similarity = tf.matmul(valid_embed, norm_embed, transpose_b=True)

        print("Beginning training procedure...")
        tf.initialize_all_variables().run()

        saver = tf.train.Saver()

        batches = self._enc.skipgram_batches(text,
                                             window_size=opts["window_size"],
                                             batch_size=opts["batch_size"],
                                             max_batches=opts["epochs"])

        # iterate through batches and keep track of average loss
        total_loss, steps = 0, 0
        last_loss = None
        for step, batch in enumerate(batches):
            feed_dict = {x: batch[0], y: batch[1]}
            _, step_loss = session.run([optimizer, loss], feed_dict=feed_dict)
            total_loss += step_loss
            steps += 1
            if step > 0 and step % opts["log_every"] == 0:
                avg_loss = total_loss / steps
                pct = 0 if last_loss is None else (avg_loss - last_loss)*100. / avg_loss
                print("Step: {:<7} Loss: {:<7.2f} ({:.2f}%)".format(step, total_loss, pct))
                last_loss = avg_loss
                total_loss, steps = 0, 0

            if step % (opts["log_every"] * 5) == 0:
                sim = similarity.eval()
                for i in range(opts["valid_size"]):
                    valid_gram = self._enc.decode_gram(x_valid[i].eval())
                    top_k = 5
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_gram
                    for k in range(top_k):
                        close_word = self._enc.decode_gram(nearest[k])
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
            final_embed = norm_embed.eval()
        save_path = saver.save(session, opts["model_file"])
        print("Model saved to: '{}'".format(save_path))


def train(text, opts):
    print("Options:", opts)
    with tf.Graph().as_default(), tf.Session() as session:
        model = Gram2Vec(text, opts, session)


def maybe_encode_bigrams(text, filename):
    if os.path.exists(filename):
        print("Found {}".format(filename))
        return data_io.GramEncoder.load(filename)
    else:
        print("Encoding...")
        enc = data_io.GramEncoder.load_text(text, 2)
        enc.save(filename)
        return enc


def main():
    text8_zip = data_io.maybe_download_text8()
    text = data_io.read_zipped_text(text8_zip)

    encoder_file = "text8_bigrams.pkl"
    model_file = "model.ckpt"

    maybe_encode_bigrams(text, encoder_file)

    opts = dict(n=2, batch_size=1024, emb_dim=128, window_size=1, neg_samples=64,
                lr=1.0, epochs=50000, valid_size=8, valid_head=100, log_every=1000,
                encoder_file=encoder_file, model_file=model_file)

    train(text, opts)


if __name__ == "__main__":
    main()
