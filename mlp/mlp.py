import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

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

class Mlp(object):
    def __init__(self, model_name, n_feat, n_hidden, n_node, n_epoch, n_train,
                 filename='feature.csv', init_b=1.0, r_l=0.1, unkn_Y = False,
                 random=False, intvl_save=100, intvl_write=10, intvl_print=10,
                 compact_plot=True, seed = 1234):
        # NP settings: print 250 chars/line; no summarization; always floats
        np.set_printoptions(linewidth=250, threshold=np.nan, suppress=True)
        # To supress Tensorflow from printing INFO
        tf.logging.set_verbosity(tf.logging.ERROR)
        df = pd.read_csv(filename, header=0, sep=',', index_col=0)
        # Randomize rows
        if(random):
            df = df.sample(frac=1, random_state=self.seed)
            df = df.reset_index(drop = True)
        self.model_name = model_name
        self.n_feat = n_feat
        self.n_node = n_node
        self.n_hidden = n_hidden
        self.n_epoch = n_epoch
        self.n_train = n_train
        self.init_b = init_b
        self.r_l = r_l
        self.intvl_save = intvl_save
        self.intvl_write = intvl_write
        self.intvl_print = intvl_print
        self.compact_plot = compact_plot
        self.seed = seed
        # Normalize features
        for i in range(n_feat):
            df.iloc[:, i] = ((df.iloc[:, i] - df.iloc[:, i].mean())
                             /df.iloc[:, i].std())
        self.df = df
        # Split data
        X = self.df.iloc[:, 0:n_feat].values
        self.X_train = X[0:n_train,]
        self.X_test = X[n_train: ,]

        Y = self.df.iloc[:, n_feat:].values
        self.Y_train = Y[0:n_train,]
        self.Y_test = Y[n_train:,]

    def new_model(self):
        # input and output layers placeholders
        X = tf.placeholder(tf.float32, [None, self.n_feat], name='X')
        Y = tf.placeholder(tf.float32, [None, 1], name='Y')

        # Weights, biases, and output function
        W = {}
        b = {}
        y = {}

        # Hidden layers construction
        for i in range(self.n_hidden):
            # hidden Layer 1: # of inputs (aka # of rows) has to be # of features
            if i == 0:
                n_in = self.n_feat
            else:
                n_in = self.n_node

            layer = 'h' + str(i+1)
            W[layer] = tf.get_variable('W'+str(i+1), shape=[n_in, self.n_node],
                                       initializer=(tf.contrib.layers
                                                    .xavier_initializer(
                                                       seed=self.seed)))
            b[layer] = tf.get_variable('b'+str(i+1),
                                        initializer=(tf.zeros([1, self.n_node])
                                                     + self.init_b))
            # Hidden layer 1: Input is X
            if i == 0:
                y[layer] = tf.nn.sigmoid(tf.matmul(X, W[layer]) + b[layer])
            # Other hidden layers: connect from its previous layer y[layer-1]
            else:
                prev_layer = 'h'+str(i)
                y[layer] = tf.nn.sigmoid(tf.matmul(y[prev_layer], W[layer])
                                         + b[layer])
        # Output layer construction
        W['out'] = tf.get_variable('Wout', shape=[self.n_node, 1],
                                   initializer=tf.contrib.layers
                                               .xavier_initializer(seed=self.seed))
        b['out'] = tf.get_variable('bout', initializer=(tf.zeros([1, 1])
                                                        + self.init_b))
        y['out'] = tf.nn.sigmoid(tf.matmul(y['h'+str(len(y))], W['out'])
                                 + b['out'])
        # Loss function: binary cross entropy with 1e-30 to avoid log(0)
        cross_entropy = -tf.reduce_sum(Y * tf.log(y['out']+1e-30)
                                       + (1-Y) * tf.log(1-y['out']+1e-30),
                                       reduction_indices=[1])
        # Back-propagation
        train_step = (tf.train.GradientDescentOptimizer(self.r_l)
                      .minimize(cross_entropy))
        # Store in class instance
        self.X = X
        self.Y = Y
        self.W = W
        self.b = b
        self.y = y
        self.cross_entropy = cross_entropy
        self.train_step = train_step
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

    def continue_model(self, meta_name, model_path='./mlp/checkpoints/'):
        # Resume from the checkpoint
        self.saver = tf.train.import_meta_graph(model_path
                                                + meta_name + '.meta')
        graph = tf.get_default_graph()

        # input and output layers placeholders
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')

        # Weights, biases, and output function
        W = {}
        b = {}
        y = {}

        # Hidden layers construction
        for i in range(self.n_hidden):
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
        train_step = (tf.train.GradientDescentOptimizer(self.r_l)
                      .minimize(cross_entropy))

        # Store in class instance
        self.X = X
        self.Y = Y
        self.W = W
        self.b = b
        self.y = y
        self.cross_entropy = cross_entropy
        self.train_step = train_step
        self.sess = tf.Session()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def predict(self, mtx_in, mtx_rst=None):
        Y_pred = self.sess.run(self.y['out'], feed_dict={self.X: mtx_in})
        Y_pred = Y_pred.round()

        if mtx_rst is not None:
            acc = tf.reduce_mean(tf.cast(tf.equal(Y_pred, mtx_rst),
                                         tf.float32))
            print('Accuracy:', self.sess.run(acc))

        return Y_pred

    def train_model(self, epoch_start):
        # './mlp/datapoints
        #   /[model_name]_[n_hidden]_[n_node]_[compact/detailed].csv'
        postfix = {True: 'compact', False: 'detailed'}
        fmt = '_'.join(['./mlp/datapoints/' + self.model_name, str(self.n_hidden),
                        str(self.n_node), postfix[self.compact_plot] + '.csv'])

        with open(fmt, 'a') as f:
            writer = csv.writer(f)
            if epoch_start == 0:
                writer.writerow(self.get_pts_csv_header())

            print()
            print("Epoch\tTraining   Testing")
            print("Number\tAccuracy   Accuracy")
            for epoch in range(epoch_start, self.n_epoch):
                acc_tr, acc_ts = None, None # reset

                if epoch % self.intvl_write == 0:
                    acc_tr, acc_ts = self.get_acc()
                    self.write_pts_csv(writer, epoch, acc_tr, acc_ts)

                if epoch % self.intvl_print == 0:
                    # calculate accuracy if it there was no write in this epoch
                    if acc_tr or acc_ts is None:
                        acc_tr, acc_ts = self.get_acc()
                    print("{}\t{:.2f}\t   {:.2f}".format(epoch, acc_tr, acc_ts))

                if epoch % self.intvl_save == 0:
                    self.saver.save(self.sess, "./mlp/checkpoints/"+self.model_name,
                               global_step = epoch)
                    print("\u001B[33m#### Session Saved @ epoch "
                          "{} ####\u001b[0m".format(epoch))

                # for every sample
                for i in range(self.X_train.shape[0]):
                    self.sess.run(self.train_step, feed_dict={self.X: self.X_train[i, None],
                                                    self.Y: self.Y_train[i, None]})
            # Save everything after last epoch
            acc_tr, acc_ts = self.get_acc()
            self.write_pts_csv(writer, self.n_epoch, acc_tr, acc_ts)
            print("{}\t{:.2f}\t   {:.2f}".format(self.n_epoch, acc_tr, acc_ts))
            self.saver.save(self.sess, "./mlp/checkpoints/"+self.model_name,
                       global_step = self.n_epoch)
            print("\u001B[33m#### Session Saved @ epoch "
                  "{} ####\u001b[0m".format(self.n_epoch))

    def get_acc(self):
        # Compute training accuracy
        Y_pred_tr = self.sess.run(self.y['out'], feed_dict={self.X: self.X_train})
        acc_tr = tf.reduce_mean(tf.cast(tf.equal(tf.round(Y_pred_tr),
                                                 self.Y_train), tf.float32))
        # Compute testing accuracy
        Y_pred_ts = self.sess.run(self.y['out'], feed_dict={self.X: self.X_test})
        acc_ts = tf.reduce_mean(tf.cast(tf.equal(tf.round(Y_pred_ts),
                                                 self.Y_test), tf.float32))
        return self.sess.run(acc_tr), self.sess.run(acc_ts)

    def get_pts_csv_header(self):

        csv_header = ['epoch']

        if self.compact_plot:
            for i in range(self.n_hidden):
                layer =  str(i+1)
                csv_header += ['W' + layer, 'b' + layer]
            csv_header += ['Wout']
            csv_header += ['bout']
        else:
            # Get names for weights in each hidden layers
            for i in range(self.n_hidden):
                # first hidden layer, # of input is # of features
                if i == 0:
                    n_in = self.n_feat
                else:
                    n_in = self.n_node
                n_out = self.n_node
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
            csv_header += ['Wout_1_' + str(i+1) for i in range(self.n_node)]
            csv_header += ['bout_1']

        csv_header += ['training_acc', 'testing_acc']

        return csv_header

    def write_pts_csv(self, writer, epoch, acc_tr, acc_ts):
        line = [epoch]  # Line to be written

        if(self.compact_plot):
            # For every hidden layer
            for i in range(self.n_hidden):
                layer = 'h' + str(i+1)
                line += [self.sess.run(self.W[layer]).flatten().mean(),
                         self.sess.run(self.b[layer]).flatten().mean()]
            # Add final layer
            line += [self.sess.run(self.W['out']).flatten().mean(),
                     self.sess.run(self.b['out']).flatten().mean()]
        else:
            # For every hidden layer
            for i in range(self.n_hidden):
                layer = 'h' + str(i+1)
                line += self.sess.run(self.W[layer]).flatten().tolist()
                line += self.sess.run(self.b[layer]).flatten().tolist()
            # Add final layer
            line += self.sess.run(self.W['out']).flatten().tolist()
            line += self.sess.run(self.b['out']).flatten().tolist()
        # Add accuracy, too
        line += [acc_tr, acc_ts]
        writer.writerow(line)
