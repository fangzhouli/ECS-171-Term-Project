# /mlp/fake\_feature

This is a directory with fake datasets that are used for testing `/mlp/mlp.py`

# Files

## genFake.R

The script that generates all of the '.csv' files under this directory. Each
feature in the files is generated using normal distribution with slightly
different mean and standard deviations. The result (i.e. the last column which
has the values of whether team A wins the game) is generated based on a
randomly picked formula below with a bit of noise:


    f = (0.5*(a_f1-b_f1) + 0.3*(a_f2-b_f2) - 0.6*(a_f3-b_f3) + 0.4*(a_f4-b_f4)
        - 0.4*(a_f5-b_f5)
        + [a random number uniformly picked between -250 ~ 250 as noise])

Then, the last column will be

    if f >= 0:
      win = 1
    else:
      win = 0

## feature.csv

A randomly generated dataset that is used as a placeholder for
'/pre_game_teams.csv' with 10K rows of data.

## input\_with\_wincol.csv

A file used as the untrained dataset to test 'mlp.py'. It is generated in the
same way as 'feature.csv' but with different random values.

## input\_no\_wincol.csv

This file is just `input_with_wincol.csv` without the `win` column.
