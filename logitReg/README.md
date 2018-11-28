# /logitReg

A directory with everything related to fitting logistic regression model with
`/pre_game_teams.csv` and `/data_processing/post_game_team_diff.csv` and computing the accuracies.

## How To Run

First, make sure R is installed and has the package `data.table` which can be
installed with the command `install.packages(data.table)` in R's interactive
window.

Then, the program can be run simply by calling the script with `rscript`, which loads the datasets mentioned above separately, uses 70% of the rows as training set, tests the model with the remaining data, and computes the accuracies.

```
$ rscript logitReg/logitReg.R
Running ./data_processing/output/post_game_team_diff.csv
Training Accuracy:	0.9488
Testing Accuracy:	0.9489
Running ./pre_game_teams.csv
Training Accuracy:	0.647
Testing Accuracy:	0.6032
$
```
