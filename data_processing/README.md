# /data_processing

A directory with everything related to data processing.     
**Note: you don't have to run these scripts if you already have the data under `output/` directory since processing data may take a long time**

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
Read all play-by-play csv files and return a data frame containing player ability scores for each player.
#### How to use:
python3 frame.py
### /data_processing/readFile.py
An API for reading csv file and returning a list containing each row of that csv file.
### /data_processing/remove_outlier.py
Take the dataframe and use Isolation forest to detect outliers and output file with name "post_game_team_diff_removed_outlier.csv" .  
#### How to use:
Pass in the path such that the list of your path contains:   

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
Pass in the path such that the list of your path contains:   

    $ ls  
    README.md                         post_game_team_diff_generator.py
    dataframe.py                      readFile.py
    features_gen.py                   remove_outlier.py
    frame.py                          tsne_graph_gen.py
    output/
Then run the program

### /pre_game_teams_gen.py
Take data from data.csv and compute expected team features and output to pre_game_teams.csv
### How to use:
In current path: python3 pre_game_teams_gen.py

### /post_game_team_diff_generator.py
Take data from RegularSeasonDetailedResults.csv and compute post-game team features and output to post_game_team_diff.csv
### How to use:
In current path: python3 post_game_team_diff_generator.py
