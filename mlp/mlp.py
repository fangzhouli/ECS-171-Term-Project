import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import atexit   # For saving weights when program exits

# Global variables
class glb:
    seed = 1234
    df = None       # dataframe
    n_feat = None   # number of features
    n_node = None   # number of neurals in each hidden layer
    n_hidden = None # number of hidden layers
    n_epoch = None  # number of training epochs
    id_2018 = None  # sampleID of the 1st row of 2018's matches in the dataset.
    init_b = None   # initial value of bias
    r_l = None      # learning rate

# Initialization of global variables.
# Input:
#   - @n_feat: int
#        Number of features
#   - @n_node: int
#        Number of neurons in a hidden layer
#   - @n_hidden: int
#        Number of hidden layers
#   - @n_epoch: int
#        Number of epochs
#   - @id_2018: int
#        sampleID of the 1st row of 2018's matches in the dataset.
#   - @init_b: float, default 1.0
#        Initial value of biases
#   - @r_l: float, default 0.1
#        Learning rate
#   - @random: boolean, default False
#        Flag for whether to randomize the rows.
#   - @filename: str, default 'feature.csv'
#        Name of the file that contains the dataset
#
# Output:
#   - N/A
def init(n_feat, n_node, n_hidden, n_epoch, id_2018,
         init_b=1.0, r_l=0.1, random=False, filename='feature.csv'):
    # NP settings: print 250 chars/line; no summarization; always print floats
    np.set_printoptions(linewidth=250, threshold=np.nan, suppress=True)
    df = pd.read_csv(filename, header=0, sep=',', index_col=0)
    # Randomize rows
    if(random):
        df = df.sample(frac=1, random_state=glb.seed)
        df = df.reset_index(drop = True)
    glb.n_feat = n_feat
    glb.n_node = n_node
    glb.n_hidden = n_hidden
    glb.n_epoch = n_epoch
    glb.id_2018 = id_2018
    glb.init_b = init_b
    glb.r_l = r_l
    # Normalize features
    for i in range(n_feat):
        df.iloc[:, i] = ((df.iloc[:, i] - df.iloc[:, i].mean())
                          /df.iloc[:, i].std())
    glb.df = df


def run():
    # Split data
    X = glb.df.iloc[:, 0:glb.n_feat].values
    Y = glb.df.iloc[:, glb.n_feat:].values
    X_train = X[0:glb.id_2018,]
    X_test = X[glb.id_2018:,]
    Y_train = Y[0:glb.id_2018,]
    Y_test = Y[glb.id_2018:,]

    # input and output layers placeholders
    X = tf.placeholder(tf.float32, [None, glb.n_feat], name='X')
    Y = tf.placeholder(tf.float32, [None, 1], name='Y')

    # Weights, biases, and output function
    W = {}
    b = {}
    y = {}

    # Hidden layers construction
    for i in range(glb.n_hidden):
        # hidden Layer 1: # of inputs (aka # of rows) has to be # of features
        if i == 0:
            n_in = glb.n_feat
        else:
            n_in = glb.n_node

        layer = 'h' + str(i+1)
        W[layer] = tf.get_variable('W'+str(i+1), shape=[n_in, glb.n_node],
                                   initializer=(tf.contrib.layers
                                                .xavier_initializer(
                                                   seed=glb.seed)))
        b[layer] = tf.get_variable('b'+str(i+1),
                                    initializer=(tf.zeros([1, glb.n_node])
                                                 + glb.init_b))

        # Hidden layer 1: Input is X
        if i == 0:
            y[layer] = tf.nn.sigmoid(tf.matmul(X, W[layer]) + b[layer])
        # Other hidden layers: connect from its previous layer y[layer-1]
        else:
            prev_layer = 'h'+str(i)
            y[layer] = tf.nn.sigmoid(tf.matmul(y[prev_layer], W[layer])
                                     + b[layer])

    # Output layer construction
    W['out'] = tf.get_variable('Wout', shape=[glb.n_node, 1],
                               initializer=tf.contrib.layers
                                           .xavier_initializer(seed=glb.seed))
    b['out'] = tf.get_variable('bout', initializer=(tf.zeros([1, 1])
                                                    + glb.init_b))
    y['out'] = tf.nn.sigmoid(tf.matmul(y['h'+str(len(y)-1)], W['out'])
                             + b['out'])

    # Loss function: binary cross entropy with 1e-30 to avoid log(0)
    cross_entropy = -tf.reduce_sum(Y * tf.log(y['out']+1e-30)
                                   + (1-Y) * tf.log(1-y['out']+1e-30),
                                   reduction_indices=[1])
    # Back-propagation
    train_step = (tf.train.GradientDescentOptimizer(glb.r_l)
                  .minimize(cross_entropy))
    init = tf.global_variables_initializer()

    # Get training and testing accuracy
    def get_acc():
        # Compute training accuracy
        Y_pred_tr = sess.run(y['out'], feed_dict={X: X_train})
        acc_tr = tf.reduce_mean(tf.cast(tf.equal(tf.round(Y_pred_tr), Y_train),
                                        tf.float32))
        # Compute testing accuracy
        Y_pred_ts = sess.run(y['out'], feed_dict={X: X_test})
        acc_ts = tf.reduce_mean(tf.cast(tf.equal(tf.round(Y_pred_ts), Y_test),
                                        tf.float32))
        return sess.run(acc_tr), sess.run(acc_ts)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(glb.n_epoch):
            print('Epoch', epoch)
            print("Accuracy:\nTraining:\t{}\nTesting:\t{}".format(*get_acc()))
            print()
            # for every sample
            for i in range(X_train.shape[0]):
                sess.run(train_step, feed_dict={X: X_train[i, None],
                                                Y: Y_train[i, None]})
            saver.save(sess, "./mlp")
        print('Epoch', glb.n_epoch)
        print("Accuracy:\nTraining:\t{}\nTesting:\t{}".format(*get_acc()))
        print()

# Test generated data 'fake_feature/feature.csv'
def test_gen():
    init(10, 10, 2, 100, 9001, filename='fake_feature/feature.csv')
    run()
