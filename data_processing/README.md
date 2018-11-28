# /data_processing

A directory with everything related to data processing.

## Sub-directories

### raw

A directory that contains the raw data that were used.

### output

A directory that contains the processed datasets except for `features.csv`,
which is at `/features.csv` .

## Files

### /data_processing/data.py

### /data_processing/dataframe.py

### /data_processing/dfeatures_gen.py

### /data_processing/frame.py

### /data_processing/readFile.py

### /data_processing/remove_outlier.py
Take the dataframe and use Isolation forest to detect outliers and output file with name "post_game_team_diff_removed_outlier.csv" .  
#### How to use:
Pass in the root path to your path such that the list of your path contains:   

    $ ls  
    README.md                         post_game_team_diff_generator.py
    dataframe.py                      readFile.py
    features_gen.py                   remove_outlier.py
    frame.py                          tsne_graph_gen.py
    output/
Then run the program

### /data_processing/tsne_graph_gen.py
Take the dataframe and use TSNE algorithm to redunce the dimension of the data to 2 and plot the graph
#### How to use:
Pass in the root path to your path such that the list of your path contains:   

    $ ls  
    README.md                         post_game_team_diff_generator.py
    dataframe.py                      readFile.py
    features_gen.py                   remove_outlier.py
    frame.py                          tsne_graph_gen.py
    output/
Then run the program
