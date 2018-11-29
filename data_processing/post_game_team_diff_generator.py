import os
import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

output_class = [1, 0]

all_games = pd.read_csv('./NCAA_data/RegularSeasonDetailedResults.csv', sep=',')

all_games['WFGM'] = all_games['WFGM'].div(all_games['WFGA'], axis = 0)*100.0
all_games['WFGM3'] = all_games['WFGM3'].div(all_games['WFGA3'], axis = 0)*100.0
all_games['WFTM'] = all_games['WFTM'].div(all_games['WFTA'], axis = 0)*100.0
all_games['LFGM'] = all_games['LFGM'].div(all_games['LFGA'], axis = 0)*100.0
all_games['LFGM3'] = all_games['LFGM3'].div(all_games['LFGA3'], axis = 0)*100.0
all_games['LFTM'] = all_games['LFTM'].div(all_games['LFTA'], axis = 0)*100.0
all_games = all_games.rename(columns = {'WFGM':'WFGP', 'WFGM3': 'WFG3P', 'WFTM': 'WFTP', 'LFGM':'LFGP', 'LFGM3': 'LFG3P', 'LFTM': 'LFTP'}).drop(['WFGA', 'WFGA3', 'WFTA', 'LFGA', 'LFGA3','LFTA'], axis = 1)

games_2003 = all_games[all_games['Season'] == 2003]
games_2004 = all_games[all_games['Season'] == 2004]
games_2005 = all_games[all_games['Season'] == 2005]
games_2006 = all_games[all_games['Season'] == 2006]
games_2007 = all_games[all_games['Season'] == 2007]
games_2008 = all_games[all_games['Season'] == 2008]
games_2009 = all_games[all_games['Season'] == 2009]
games_2010 = all_games[all_games['Season'] == 2010]


#pre-process data
if not os.path.isfile('./data_processing/output/post_game_team_diff.csv'):
	#training and testing set
	games_from_2003_to_2010 = []
	for games in [games_2003, games_2004, games_2005, games_2006, games_2007, games_2008, games_2009, games_2010]:
		for i in range(games.shape[0]):
			diff_stats = np.array(games.iloc[i,8:18])-np.array(games.iloc[i,18:])
			games_from_2003_to_2010.append(diff_stats.tolist()+[output_class[0]])
			games_from_2003_to_2010.append((-1 * diff_stats).tolist()+[output_class[1]])
				
	games_from_2003_to_2010 = pd.DataFrame(games_from_2003_to_2010, columns = ['FG%_diff', '3P%_diff', 'FT%_diff', 'OR_diff', 'DR_diff', 'AST_diff', 'TO_diff', 'STL_diff','BLK_diff', 'PF_diff', 'W/L']).dropna()
	games_from_2003_to_2010.to_csv('./data_processing/output/post_game_team_diff.csv', sep=',')

