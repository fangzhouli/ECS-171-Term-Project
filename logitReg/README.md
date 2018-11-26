# /logitReg

A directory with everything related to fitting logistic regression model with
`features.csv` and computing its accuracy.

## How To Run

First, make sure R is installed and has the package `data.table` which can be
installed with the command `install.packages(data.table)` in R's interactive
window.

Then, the program can be run with the following which trains with data from
2011 to 2017 and tests the model with data from 2018:

``` R
$ R
> source('logitReg/logitReg.R')
> run(n_train=40113, random=FALSE)
Training Accuracy:	0.647
Testing Accuracy:	0.6032
```
