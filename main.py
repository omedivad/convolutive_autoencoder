from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

tf.summary.FileWriterCache.clear()

## importo il MNIST (ogni immagine è 28x28)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parametri ottimizzazione
learning_rate = 0.0006
batch_size = 100
n_iteration = 50
display_step = 1000

# layer di input
X = tf.placeholder(tf.float32, [None, 784])
x = tf.reshape(X, [-1, 28, 28, 1])


def encoder(x):
    with tf.name_scope("encoder"):
        #encoder 4 hidden layer
        layer_1 = tf.layers.conv2d(
            inputs = x,
            filters = 1,
            kernel_size = [2, 2],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        # [27, 27, 32]
        layer_2 = tf.layers.conv2d(
            inputs = layer_1,
            filters = 64,
            kernel_size=[5, 5],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        layer_3 = tf.layers.conv2d(
            inputs=layer_2,
            filters = 32,
            kernel_size=[5, 5],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        # dimensione 18 x 18 x 20

        layer_4 = tf.layers.conv2d(
            inputs=layer_3,
            filters=1,
            kernel_size=[5, 5],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        return layer_4


def decoder(z):
    with tf.name_scope("decoder"):

        layer_4 = tf.layers.conv2d_transpose(
            inputs=z,
            filters=1,
            kernel_size=[5, 5],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        layer_3 = tf.layers.conv2d_transpose(
            inputs=layer_4,
            filters = 32,
            kernel_size=[5, 5],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        layer_2 = tf.layers.conv2d_transpose(
            inputs = layer_3,
            filters = 64,
            kernel_size=[5, 5],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        layer_1 = tf.layers.conv2d_transpose(
            inputs = layer_2,
            filters=1,
            kernel_size=[2, 2],
            strides=(1, 1),
            activation=tf.nn.relu,
        )

        return layer_1

# modello
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# predizione (y_true è l'ingresso)
y_pred = decoder_op
y_true = x

# loss e ottimizzazione
with tf.name_scope("QME"):
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# inizializza le variabili
init = tf.global_variables_initializer()

# salvo le variabili
saver = tf.train.Saver()

# inizio della sessione

with tf.Session() as sess:
    # Restore variables from disk.
    # saver.restore(sess, "/home/davide/Scrivania/temp/model.ckpt")

    # scrivi per tensorboard
    merged = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('/home/davide/Scrivania/temp/logs', sess.graph)

    sess.run(init)

    # train
    for i in range(1, n_iteration + 1):
        batch, _ = mnist.train.next_batch(batch_size)

        _, l = sess.run([optimizer, loss], feed_dict={X: batch})

        print('Step %i: Minibatch Loss: %f' % (i, l))


    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")

    plt.show()

    save_path = saver.save(sess, "/home/davide/Scrivania/temp/model.ckpt")
    print("Model saved in path: %s" % save_path)