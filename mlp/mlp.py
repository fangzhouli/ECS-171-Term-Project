import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import os
import sys

# Global variables
class glb:
    seed = 1234
    df = None       # dataframe
    n_feat = None   # number of features
    n_node = None   # number of neurals in each hidden layer
    n_hidden = None # number of hidden layers
    n_epoch = None  # number of training epochs
    n_train = None  # number of rows as training set
    init_b = None   # initial value of bias
    r_l = None      # learning rate
    X_train = None  # Training inputs
    X_test = None   # Testing inputs
    Y_train = None  # Training outputs
    Y_test = None   # Testing outputs

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
#   - @n_train: int
#        number of rows from the beginning to be used as the training set.
#   - @init_b: float, default 1.0
#        Initial value of biases
#   - @r_l: float, default 0.1
#        Learning rate
#   - @unkn_Y: boolean, default False
#        flag to indicate whether Y is known, i.e., if there's an actual result
#        column.
#   - @random: boolean, default False
#        Flag for whether to randomize the rows.
#   - @filename: str, default 'feature.csv'
#        Name of the file that contains the dataset
#
# Output:
#   - N/A
def init(n_feat, n_node, n_hidden, n_epoch, n_train, init_b=1.0, r_l=0.1,
         unkn_Y = False, random=False, filename='feature.csv'):
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
    glb.n_train = n_train
    glb.init_b = init_b
    glb.r_l = r_l
    # Normalize features
    for i in range(n_feat):
        df.iloc[:, i] = ((df.iloc[:, i] - df.iloc[:, i].mean())
                          /df.iloc[:, i].std())
    glb.df = df
    # Split data
    X = glb.df.iloc[:, 0:n_feat].values
    glb.X_train = X[0:n_train,]
    glb.X_test = X[n_train: ,]

    # Assign them if we do have actual results
    if unkn_Y == False:
        Y = glb.df.iloc[:, n_feat:].values
        glb.Y_train = Y[0:n_train,]
        glb.Y_test = Y[n_train:,]
    else:
        glb.Y_train = None
        glb.Y_test = None

# Create a new model through training with data from 2011~2017
# Input:
#   - @model_name: str
#        Prefix of the name of the model's files to be saved in
#          './mlp/checkpoints' and './mlp/checkpoints/'.
#   - @intvl_save: int, default 100
#        Number of epochs to run before saving Tensorflow files
#   - @intvl_write: int, default 10
#        Number of epochs to run before saving to './mlp/datapoints/*.csv'
#   - @intvl_print: int, default 10
#        Number of epochs to run before printing accuracy
#   - @compact_plot: boolean, default True
#        flag for whether to plot compact or detailed plot; the difference
#        between a compact and a detailed plot is that former contains mean of
#        each layer's weights and biases and the latter contains individual
#        value of every single weight.
# Output:
#   - Tensorflow files saved under './mlp/checkpoints'.
def new_model(model_name, intvl_save=100, intvl_write=10, intvl_print=10,
              compact_plot=True):
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
    y['out'] = tf.nn.sigmoid(tf.matmul(y['h'+str(len(y))], W['out'])
                             + b['out'])

    # Loss function: binary cross entropy with 1e-30 to avoid log(0)
    cross_entropy = -tf.reduce_sum(Y * tf.log(y['out']+1e-30)
                                   + (1-Y) * tf.log(1-y['out']+1e-30),
                                   reduction_indices=[1])
    # Back-propagation
    train_step = (tf.train.GradientDescentOptimizer(glb.r_l)
                  .minimize(cross_entropy))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=None)

    # './mlp/datapoints
    #   /[model_name]_[n_hidden]_[n_node]_[compact/detailed].csv'
    postfix = {True: 'compact', False: 'detailed'}
    fmt = '_'.join(['./mlp/datapoints/' + model_name, str(glb.n_hidden),
                    str(glb.n_node), postfix[compact_plot] + '.csv'])

    with tf.Session() as sess, open(fmt, 'a') as f:
        sess.run(init)
        writer = csv.writer(f)
        writer.writerow(get_pts_csv_header(compact_plot))
        for epoch in range(glb.n_epoch):
            acc_tr, acc_ts = None, None # reset

            if epoch % intvl_write == 0:
                acc_tr, acc_ts = get_acc(sess, X, Y, y)
                write_pts_csv(compact_plot, sess, writer, epoch, W, b,
                              acc_tr, acc_ts)

            if epoch % intvl_save == 0:
                saver.save(sess, "./mlp/checkpoints/"+model_name,
                           global_step = epoch)
                print('Session saved.\n')

            if epoch % intvl_print == 0:
                # calculate accuracy if it there was no write in this epoch
                if acc_tr or acc_ts is None:
                    acc_tr, acc_ts = get_acc(sess, X, Y, y)
                print_acc(epoch, acc_tr, acc_ts)

            # for every sample
            for i in range(glb.X_train.shape[0]):
                sess.run(train_step, feed_dict={X: glb.X_train[i, None],
                                                Y: glb.Y_train[i, None]})
        # Save everything after last epoch
        acc_tr, acc_ts = get_acc(sess, X, Y, y)
        write_pts_csv(compact_plot, sess, writer, glb.n_epoch, W, b,
                      acc_tr, acc_ts)
        saver.save(sess, "./mlp/checkpoints/"+model_name,
                   global_step = glb.n_epoch)
        print('Session saved.\n')
        print_acc(glb.n_epoch, acc_tr, acc_ts)

# Load from the lastest model from './mlp/checkpoints/ and continue training'
# Input:
#   - @model_name: str
#        Prefix of the name of the model's file to be saved in
#          './mlp/checkpoints'.
#   - @meta_name: str
#        Prefix of the '.meta' file to be loaded.
#        E.X.: 'model-100' if the '.meta' file is named 'model-100.meta'
#   - @epoch_start: int
#        Start epoch; pretty much the end epoch of the model to be loaded, so
#          'epoch_start' should be 300 if the model to be loaded was saved at
#          epoch 300 unless user intended to do other testing.
#   - @intvl_save: int, default 100
#        Number of epochs to run before saving Tensorflow files
#   - @intvl_write: int, default 10
#        Number of epochs to run before saving to './mlp/datapoints/*.csv'
#   - @intvl_print: int, default 10
#        Number of epochs to run before printing accuracy
#   - @model_path: str, default './mlp/checkpoints/'
#        Path to the checkpoint directory.
#   - @compact_plot: boolean, default True
#        flag for whether to plot compact or detailed plot; the difference
#        between a compact and a detailed plot is that former contains mean of
#        each layer's weights and biases and the latter contains individual
#        value of every single weight.
# Output:
#   - Tensorflow files saved under './mlp/checkpoints'.
def continue_model(model_name, meta_name, epoch_start,
                   intvl_save=100, intvl_write=10, intvl_print=10,
                   model_path='./mlp/checkpoints/', compact_plot = True):
    # Resume from the checkpoint
    saver = tf.train.import_meta_graph(model_path + meta_name + '.meta')
    graph = tf.get_default_graph()

    # input and output layers placeholders
    X = graph.get_tensor_by_name('X:0')
    Y = graph.get_tensor_by_name('Y:0')

    # Weights, biases, and output function
    W = {}
    b = {}
    y = {}

    # Hidden layers construction
    for i in range(glb.n_hidden):
        layer = 'h' + str(i+1)
        W[layer] = graph.get_tensor_by_name('W' + str(i+1) + ':0')
        b[layer] = graph.get_tensor_by_name('b' + str(i+1) + ':0')

        # Hidden layer 1: Input is X
        if i == 0:
            y[layer] = tf.nn.sigmoid(tf.matmul(X, W[layer]) + b[layer])
        # Other hidden layers: connect from its previous layer y[layer-1]
        else:
            prev_layer = 'h'+str(i)
            y[layer] = tf.nn.sigmoid(tf.matmul(y[prev_layer], W[layer])
                                     + b[layer])

    # Output layer construction
    W['out'] = graph.get_tensor_by_name('Wout:0')
    b['out'] = graph.get_tensor_by_name('bout:0')
    y['out'] = tf.nn.sigmoid(tf.matmul(y['h'+str(len(y))], W['out'])
                          + b['out'])

    # Loss function: binary cross entropy with 1e-30 to avoid log(0)
    cross_entropy = -tf.reduce_sum(Y * tf.log(y['out']+1e-30)
                                   + (1-Y) * tf.log(1-y['out']+1e-30),
                                   reduction_indices=[1])
    # Back-propagation
    train_step = (tf.train.GradientDescentOptimizer(glb.r_l)
                  .minimize(cross_entropy))

    # './mlp/datapoints
    #   /[model_name]_[n_hidden]_[n_node]_[compact/detailed].csv'
    postfix = {True: 'compact', False: 'detailed'}
    fmt = '_'.join(['./mlp/datapoints/' + model_name, str(glb.n_hidden),
                    str(glb.n_node), postfix[compact_plot] + '.csv'])

    with tf.Session() as sess, open(fmt, 'a') as f:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        writer = csv.writer(f)

        for epoch in range(epoch_start, glb.n_epoch):
            acc_tr, acc_ts = None, None # reset

            if epoch % intvl_write == 0:
                acc_tr, acc_ts = get_acc(sess, X, Y, y)
                write_pts_csv(compact_plot, sess, writer, epoch, W, b,
                              acc_tr, acc_ts)

            if epoch % intvl_save == 0:
                saver.save(sess, "./mlp/checkpoints/"+model_name,
                           global_step = epoch)
                print('Session saved.\n')

            if epoch % intvl_print == 0:
                # calculate accuracy if it there was no write in this epoch
                if acc_tr or acc_ts is None:
                    acc_tr, acc_ts = get_acc(sess, X, Y, y)
                print_acc(epoch, acc_tr, acc_ts)

            # for every sample
            for i in range(glb.X_train.shape[0]):
                sess.run(train_step, feed_dict={X: glb.X_train[i, None],
                                                Y: glb.Y_train[i, None]})
        # Save everything after last epoch
        acc_tr, acc_ts = get_acc(sess, X, Y, y)
        write_pts_csv(compact_plot, sess, writer, glb.n_epoch, W, b,
                      acc_tr, acc_ts)
        saver.save(sess, "./mlp/checkpoints/"+model_name,
                   global_step = glb.n_epoch)
        print('Session saved.\n')
        print_acc(glb.n_epoch, acc_tr, acc_ts)

# Load from the lastest model from 'model_path' and make predictions on 'X'.
#   Accuracy will be given if 'Y' is not None.
# Input:
#   - @model_name: str
#        Prefix of the name of the model's file to be saved in 'ckpt_path'
#   - @meta_name: str
#        Prefix of the '.meta' file to be loaded.
#        E.X.: 'model-100' if the '.meta' file is named 'model-100.meta'
#   - @mtx_in: np.matrix
#        Matrix with input features
#   - @mtx_rst: np.matrix
#        1-D np.matrix with actual output. None if Y is unknown
#   - @ckpt_path: str, default './mlp/checkpoints/'
#        Path to the checkpoint directory.
# Output:
#   - prediction matrix from the model
def predict_from_model(model_name, meta_name, mtx_in, mtx_rst,
                       ckpt_path='./mlp/checkpoints/'):
    # Weights, biases, and output function
    W = {}
    b = {}
    y = {}

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt_path + meta_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        graph = tf.get_default_graph()

        # input and output layers placeholders
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')

        # Hidden layers construction
        for i in range(glb.n_hidden):
            layer = 'h' + str(i+1)
            W[layer] = graph.get_tensor_by_name('W' + str(i+1) + ':0')
            b[layer] = graph.get_tensor_by_name('b' + str(i+1) + ':0')

            # Hidden layer 1: Input is X
            if i == 0:
                y[layer] = tf.nn.sigmoid(tf.matmul(X, W[layer]) + b[layer])
            # Other hidden layers: connect from its previous layer y[layer-1]
            else:
                prev_layer = 'h'+str(i)
                y[layer] = tf.nn.sigmoid(tf.matmul(y[prev_layer], W[layer])
                                         + b[layer])

        # Output layer construction
        W['out'] = graph.get_tensor_by_name('Wout:0')
        b['out'] = graph.get_tensor_by_name('bout:0')
        y['out'] = tf.nn.sigmoid(tf.matmul(y['h'+str(len(y))], W['out'])
                              + b['out'])

        Y_pred = (sess.run(y['out'], feed_dict={X: glb.X_test})).round()

        if mtx_rst is not None:
            acc = tf.reduce_mean(tf.cast(tf.equal(Y_pred,glb.Y_test),
                                         tf.float32))
            print('Accuracy:', sess.run(acc))

    tf.reset_default_graph()

    return Y_pred

# Get training and testing accuracy
# Input:
#   -@sess: tf.Session()
#       The current session
#   -@X: tf.placeholder
#       Input placeholder
#   -@Y: tf.placeholder
#       Output placeholder
#   -@y: list
#       The list of output functions
def get_acc(sess, X, Y, y):
    # Compute training accuracy
    Y_pred_tr = sess.run(y['out'], feed_dict={X: glb.X_train})
    acc_tr = tf.reduce_mean(tf.cast(tf.equal(tf.round(Y_pred_tr),
                                             glb.Y_train), tf.float32))
    # Compute testing accuracy
    Y_pred_ts = sess.run(y['out'], feed_dict={X: glb.X_test})
    acc_ts = tf.reduce_mean(tf.cast(tf.equal(tf.round(Y_pred_ts),
                                             glb.Y_test), tf.float32))
    return sess.run(acc_tr), sess.run(acc_ts)

# Print accuracy
# Input:
#   -@epoch: int
#       The current epoch
#   -@acc_tr: float
#       Calculated training accuracy
#   -@acc_ts: float
#       Calculated testing accuracy
# Output:
#   - Printed accuracy
def print_acc(epoch, acc_tr, acc_ts):
    print('Epoch', epoch)
    print("Accuracy:")
    print("Training:\t{:.2f}".format(acc_tr))
    print("Testing:\t{:.2f}".format(acc_ts))


# Create a header consists of each weight for './mlp/datapoints/*.csv' files
#   Compact Plot:
#     Each column is either mean of weights or biases of each layer
#   Detailed Plot:
#     Each column is either the value of each individual weight and bias with
#     with the following format:
#       Weights: 'W[layer #]_[destination neuron #]_[origin neuron #]'
#       Bias: 'b[layer #]_[neuron #]
#       Example:
#         epoch W1_1_1 W1_2_1 W1_3_1 ... training_acc testing_acc
# Input:
#   - @compact_plot: boolean,
#        flag for whether to plot compact or detailed plot
# Output:
#   - a list with column names of the weights, biases, and accuracy
def get_pts_csv_header(compact_plot):
    # The algorithm might look hedious, and there really isn't an easy way to
    #   explain it soley with comments, but running this snippet line by line
    #   will make it quite obvious.
    csv_header = ['epoch']

    if compact_plot:
        for i in range(glb.n_hidden):
            layer =  str(i+1)
            csv_header += ['W' + layer, 'b' + layer]
        csv_header += ['Wout']
        csv_header += ['bout']
    else:
        # Get names for weights in each hidden layers
        for i in range(glb.n_hidden):
            # first hidden layer, # of input is # of features
            if i == 0:
                n_in = glb.n_feat
            else:
                n_in = glb.n_node
            n_out = glb.n_node
            # Prefix, i.e., 'W1_' for hidden layer 1, 'W2_' for hidden layer 2,
            #   etc
            pfix = ['W' + str(i+1) + '_'] * (n_in*n_out)
            # Column indice in the matrix, i.e., '1_' for column 1
            c = [str(i+1) + '_' for i in range(n_out)] * n_in
            # Row indice in the matrix, i.e., '1_' for row 1
            r = [[str(i+1)] * n_out for i in range(n_in)]
            r = [i for l in r for i in l]   # to flatten the list
            # Combine prefixes and column & row indice together.
            csv_header += [a+b+c for a,b,c in zip(pfix,c,r)]
            # Add biases to the list
            csv_header += ['b' + str(i+1) + '_'
                            + str(j+1) for j in range(n_out)]

        # Add the names for the final layer
        csv_header += ['Wout_1_' + str(i+1) for i in range(glb.n_node)]
        csv_header += ['bout_1']

    csv_header += ['training_acc', 'testing_acc']

    return csv_header

# Write weights, biases, and accuracy to './mlp/datapoints/*.csv' from current
#   session.
# Input:
#   - @compact_plot: boolean
#        flag for whether to plot compact or detailed plot
#   - @sess: tf.Session()
#        Current Tensorflow session.
#   - @writer: csv.writer
#        csv writer created for the desinated file
#   - @epoch: int
#        current epoch
#   - @W: dict
#        The weight dictionary
#   - @b: dict
#        The bias dictionary
#   - @acc_tr:
#        Training accruacy of current iteration
#   - @acc_ts:
#        Testing accuracy of current iteration
# Output:
#   - writes line to buffer which will then be stored to the specified file.
def write_pts_csv(compact_plot, sess, writer, epoch, W, b, acc_tr, acc_ts):
    line = [epoch]  # Line to be written

    if(compact_plot):
        # For every hidden layer
        for i in range(glb.n_hidden):
            layer = 'h' + str(i+1)
            line += [sess.run(W[layer]).flatten().mean(),
                     sess.run(b[layer]).flatten().mean()]
        # Add final layer
        line += [sess.run(W['out']).flatten().mean(),
                 sess.run(b['out']).flatten().mean()]
    else:
        # For every hidden layer
        for i in range(glb.n_hidden):
            layer = 'h' + str(i+1)
            line += [sess.run(W[layer]).flatten().tolist(),
                     sess.run(b[layer]).flatten().tolist()]
        # Add final layer
        line += [sess.run(W['out']).flatten().tolist(),
                 sess.run(b['out']).flatten().tolist()]
    # Add accuracy, too
    line += [acc_tr, acc_ts]
    writer.writerow(line)


# Plot the weights and accruacy of a .csv file in './mlp/datapoints/' and save
#   the plots under './mlp/plots/'
# Input:
#   - filepath: str
#       path to the .csv file to be plotted.
#       E.X.: './mlp/datapoints/fake_model_2_10_compact.csv'
# Output:
#   - plots saved under './mlp/plots/'
def plot_pts_csv(filepath):
    df = pd.read_csv(filepath, header=0, sep=',', index_col=0)
    ncol = df.shape[1]
    w = df.iloc[:, 0:(ncol-2)]
    acc = df.iloc[:, [-2 ,-1]]
    # Get file name
    name = filepath.split('/')[-1]
    name = name.split('.')[0]

    # Save weight plot
    plt.figure()
    w.plot()
    plt.legend(loc='upper left')
    plt.savefig('./mlp/plots/' + name + '_weights.png')
    # Save accuracy plot
    acc.plot()
    plt.legend(loc='upper left')
    plt.savefig('./mlp/plots/' + name + '_accuracy.png')

############################## Testing functions ##############################

# Test 'new_model()' using generated data 'fake_feature/feature.csv'.
def test_new_model():
    # MUST RUN FROM TOP DIRECTORY, I.E. YOU'RE RUNNING THIS SCRIPT USING PATH
    #   './mlp/mlp.py'.
    try:
        init(10, 10, 2, 13, 9001, filename='./mlp/fake_feature/feature.csv')
    except Exception as e:
        print(e)
        msg = (
            "\u001b[31mPlease check if you are running the script from the "
            "top directory, i.e., make sure you are running using the path"
            "'./mlp/mlp.py'. \u001b[0m")
        print(msg)
        exit(0)
    new_model('fake_model', intvl_save=2, intvl_write=2, intvl_print=1)

# Assuming 'test_gen()' is invoked beforehand and left off at epoch 13
def test_continue_model():
    init(10, 10, 2, 26, 9001, filename='./mlp/fake_feature/feature.csv')
    continue_model('fake_model', 'fake_model-13', 14,
                   intvl_save=2, intvl_write=2, intvl_print=1)

# Test 'test_predict_from_model()' using generated data
#   './mlp/fake_feature/input_*.csv'
def test_predict_from_model():
    # Test with known actual results
    init(10, 10, 2, 26, 9001, unkn_Y = False,
         filename='./mlp/fake_feature/input_with_wincol.csv')
    mtx_kn = predict_from_model('fake_model', 'fake_model-14',
                                glb.X_test, glb.Y_test,
                                ckpt_path='./mlp/checkpoints/')

    # Test with unknown actual results
    init(10, 10, 2, 26, 9001, unkn_Y = True,
         filename='./mlp/fake_feature/input_no_wincol.csv')
    mtx_ukn = predict_from_model('fake_model', 'fake_model-14',
                                 glb.X_test, None,
                                 ckpt_path='./mlp/checkpoints/')
    assert(np.all(mtx_ukn == mtx_kn))
