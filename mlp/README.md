# /mlp

A directory with everything related to using multi layer perceptrons for binary classification for this project using Tensorflow package, including construction, training, testing, and plotting.

Description of each sub-directory is in `README.md` under its directory, and
a copy of directory descriptions is also provided in the end of this file.

**Important: Must be run from (or have default working directory set to) the top directory, i.e., you get the following when you type the command in shell:**

    $ pwd
    /[YOUR_DIRECTORY_PATH]/ECS-171-Group-Project
    $ ls
    data.csv  dataframe.py  frame.py  images_RM  mlp  readFile.py

## /mlp/mlp.py

### Features

- Construct MLP with user defined number of input features, layer neurons,
hidden layers, training epochs, rows from the dataset to use as the training
set.

- Save Tensorflow graphs and variables as Tensorflow checkpoint files.

- Save weights, biases, and training & testing accuracy in CSV format

- Continue training from previously saved model

- Plot saved weights, biases, and accuracy v.s. iteration

- Load previously saved model and make prediction

- Grid search on multiple CSIF computers simultaneously

### Example: /mlp/fake\_feature/feature.csv

This example uses a randomly generated dataset with 10K rows and that has the following columns:

    sampleID a_f1 a_f2 a_f3 a_f4 a_f5 b_f1 b_f2 b_f3 b_f4 b_f5 win

As shown here, the dataset has 10 input features and a binary output. In this
example, we will train a binary classification MLP with 2 hidden layers and 10
neurons in each. First, we make sure we are in the top directory

    $ pwd
    /[YOUR_DIRECTORY_PATH]/ECS-171-Group-Project
    $ ls
    data.csv  dataframe.py  frame.py  images_RM  mlp  readFile.py

We can then import the module and initialize the attributes with

    >>> python
    >>> from mlp.mlp import *
    >>> m1 = Mlp('fake_model', 10, 2, 10, 13, 9000,
    ...          pathToDataset='./mlp/fake_feature/feature.csv',
    ...          intvl_save=4, intvl_write=2, intvl_print=1)

The last command is the same as

    >>> m1 = Mlp(model_name='fake_model', n_feat=10, n_hidden=2, n_node=10,
    ...          n_epoch=13, n_train=9000,
    ...          pathToDataset='./mlp/fake_feature/feature.csv', init_b=1.0,
    ...          r_l=0.1, random=False, intvl_save=4, intvl_write=2,
    ...          intvl_print=1, compact_plot=True, seed = 1234)

Then, we can construct the model from scratch with

    >>> m1.new_model()

Now we can start training with

    >>> m1.train_model(epoch_start=0)

    Epoch	Training   Testing
    Number	Accuracy   Accuracy
    0	0.51	   0.50
    #### Session Saved @ epoch 0 ####
    1	0.92	   0.91
    2	0.92	   0.91
    3	0.92	   0.91
    4	0.92	   0.91
    #### Session Saved @ epoch 4 ####
    5	0.92	   0.91
    6	0.92	   0.91
    7	0.92	   0.91
    8	0.92	   0.91
    #### Session Saved @ epoch 8 ####
    9	0.92	   0.91
    10	0.92	   0.91
    11	0.92	   0.91
    12	0.92	   0.91
    #### Session Saved @ epoch 12 ####
    13	0.92	   0.91
    #### Session Saved @ epoch 13 ####

Since we have `intvl_save=2, intvl_write=2, intvl_print=1`, model is saved as
Tensorflow checkpoint files every 2 epochs; weights, biases, and accuracy are saved in CSV format every 2 epochs; and training & testing accuracy is printed every 1 epoch. By default, Tensorflow checkpoint files are saved under `/mlp/checkpoints/` and CSV files are saved under `/mlp/datapoints/`.

If we decide to continue training to have a total of 26 epochs. We can use `continue_model()`:

    >>> m1.n_epoch = 26   # set new number of epochs
    >>> m1.continue_model('fake_model-13')
    >>> m1.train_model(epoch_start=14)

Now let's say we've exited the program, then we will have to form the structure
again:

    >>> m2 = Mlp('fake_model', 10, 2, 10, 26, 9000,
    ...          filename='./mlp/fake_feature/feature.csv',
    ...          intvl_save=4, intvl_write=2, intvl_print=1)
    >>> m2.continue_model('fake_model-13')
    >>> m2.train_model(epoch_start=14)

    Epoch	Training   Testing
    Number	Accuracy   Accuracy
    14	0.92	   0.91
    15	0.92	   0.90
    16	0.92	   0.90
    #### Session Saved @ epoch 16 ####
    17	0.92	   0.90
    18	0.92	   0.90
    19	0.92	   0.90
    20	0.92	   0.90
    #### Session Saved @ epoch 20 ####
    21	0.92	   0.90
    22	0.92	   0.90
    23	0.92	   0.90
    24	0.92	   0.90
    #### Session Saved @ epoch 24 ####
    25	0.92	   0.90
    26	0.92	   0.90
    #### Session Saved @ epoch 26 ####

*Note: This only works with starting from the latest checkpoint. To continue
from other checkpoints, "model\_checkpoint\_path" and
"all\_model\_checkpoint\_paths" in `/mlp/checkpoints/checkpoint` have to be
set to the corresponding checkpoint filename.*

Now that we are finished training, we can plot the weights and accuracy saved
in the CSV file under `/mlp/datapoints` with

    >>> plot_pts_csv('./mlp/datapoints/fake_model_2_10_compact.csv')

and the plots for accuracy and mean weights (shown below) can be found under
`/mlp/plots/` after running all of the code above.

![acc](/images_RM/fake_model_2_10_compact_accuracy.png)

![wgt](/images_RM/fake_model_2_10_compact_weights.png)


### Parallel Grid Search

A grid search that covers from 1 to 3 layers and 10 to 25 neurons with a step
of 5 can be done with `parallel_csif_grid_search()` which invokes 12 instances
of terminals that connect to CSIF simultaneously and perform new model
construction code above. **This function requires user to have done the
following three things beforehand in order for it to work:**

1. Have the repository cloned onto a CSIF computer

2. Have setup keyless login. See 'https://goo.gl/9xFJTA'

3. Make sure the computers that are going to be used are functional.
   Check 'https://goo.gl/fa7jS7' for which CSIF computers are up.

After finishing these tasks, see more details in `mlp/mlp.py` for descriptions
for input arguments.

## Sub-directories

### /mlp/checkpoints

This is a directory to save Tensorflow graphs and variables using Tensorflow
checkpoint format generated by `/mlp/mlp.py`

### /mlp/datapoints

This is a directory to store weights, biases, and training & testing accuracy
of every specified epoch interval of the MLP in CSV format generated by
`/mlp/mlp.py` for plotting purposes.


### /mlp/fake\_feature

This is a directory with fake datasets that are used for testing `/mlp/mlp.py`

### /mlp/plots

This is a directory to store generated plots for `/mlp/mlp.py`.
