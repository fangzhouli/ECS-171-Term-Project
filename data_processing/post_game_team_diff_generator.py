import os
import csv
import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils
from keras.optimizers import Adam

output_class = [1, 0]

all_games = pd.read_csv('/Users/leomailbox/Desktop/NCAA_data/RegularSeasonDetailedResults.csv', sep=',')

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
games_2011 = all_games[all_games['Season'] == 2011]
games_2012 = all_games[all_games['Season'] == 2012]
games_2013 = all_games[all_games['Season'] == 2013]
games_2014 = all_games[all_games['Season'] == 2014]
games_2015 = all_games[all_games['Season'] == 2015]
games_2016 = all_games[all_games['Season'] == 2016]
games_2017 = all_games[all_games['Season'] == 2017]


#pre-process data
if not os.path.isfile('/Users/leomailbox/Desktop/NCAA_data/post_game_team_diff.csv'):
		with open('/Users/leomailbox/Desktop/NCAA_data/post_game_team_diff.csv', 'w') as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(['FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'TO', 'STL','BLK', 'PF', 'W/L'])

            #training set
            games_from_2003_to_2007 = []
			for games in [games_2003, games_2004, games_2005, games_2006, games_2007]:
				for i in range(games.shape[0]):
					diff_stats = np.array(games.iloc[i,8:18])-np.array(games.iloc[i,18:])
					games_from_2003_to_2007.append(diff_stats.tolist()+[output_class[0]])
					games_from_2003_to_2007.append((-1 * diff_stats).tolist()+[output_class[1]])
					writer.writerow(diff_stats.tolist()+[output_class[0]])
					writer.writerow((-1 * diff_stats).tolist()+[output_class[1]])

				
			games_from_2003_to_2007 = pd.DataFrame(games_from_2003_to_2007, columns = ['FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'TO', 'STL','BLK', 'PF', 'W/L']).dropna()
			train_features = games_from_2003_to_2007[['FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'TO', 'STL','BLK', 'PF']].values
			train_labels = np_utils.to_categorical(games_from_2003_to_2007['W/L'].tolist(), 2)

			#testing set
			games_from_2008_to_2010 = []
			for games in [games_2008, games_2009, games_2010]:
				for i in range(games.shape[0]):
					diff_stats = np.array(games.iloc[i,8:18])-np.array(games.iloc[i,18:])
					games_from_2008_to_2010.append(diff_stats.tolist()+[output_class[0]])
					games_from_2008_to_2010.append((-1 * diff_stats).tolist()+[output_class[1]])
					writer.writerow(diff_stats.tolist()+[output_class[0]])
					writer.writerow((-1 * diff_stats).tolist()+[output_class[1]])

			games_from_2008_to_2010 = pd.DataFrame(games_from_2008_to_2010, columns = ['FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'TO', 'STL','BLK', 'PF', 'W/L']).dropna()
			test_features = games_from_2008_to_2010[['FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'TO', 'STL','BLK', 'PF']].values
			test_labels = np_utils.to_categorical(games_from_2008_to_2010['W/L'].tolist(), 2)

			#set up logistic regression model
			'''model = Sequential()
			model.add(Dense(2, use_bias = False, activation='softmax', input_dim=10))  # input dimension = number of features your data has
			model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
			model.fit(train_features, train_labels, shuffle=True, epochs=50, validation_split=0.3)
			print(model.evaluate(test_features, test_labels, verbose = 1))'''



