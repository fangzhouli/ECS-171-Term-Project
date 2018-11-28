#input: data.csv
#output: features.csv
import os
import csv
import numpy as np
import pandas as pd

path = './NCAA_data/'
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

def get_data():
	'''
	process input data, 'time' records the avg game time for each player.
	Other stats columns record stat per game for that player
	'''
	data = pd.read_csv(path+'data.csv', sep=',')
	data = data.drop(['sub_out', 'timeout_tv', 'timeout'], axis=1)
	data.iloc[:,2:-1] = data.iloc[:,2:-1].apply(pd.to_numeric)
	data['time'] = data['time']/60
	data.iloc[:,2:-1] = data.iloc[:,2:-1].div(data['n_match'],axis=0)
	TEAMS = data.loc[data['name'] == 'TEAM']['player_ID'].tolist()
	return data, TEAMS

def player_stats_gen(player_IDs, player_names_this_year, player_IDs_this_year, player_names_last_year,
	player_IDs_last_year, data, new_player_stats):
	player_stats =[]
	names = []
	for ID in player_IDs:
		names.append(player_names_this_year[player_IDs_this_year.index(ID)])
	new_names = list(set(names)-set(player_names_last_year))
	old_names = list(set(names)-set(new_names))
	if len(old_names) == 0:
		for new_name in new_names:
			player_stats.append(new_player_stats)
	else:
		for old_name in old_names:
			old_name_stats = data.iloc[player_IDs_last_year[player_names_last_year.index(old_name)]-600000-1,:].tolist()
			player_stats.append(old_name_stats[2:-1])
		for new_name in new_names:
			player_stats.append(new_player_stats)

	return player_stats

def team_stats_gen(player_stats):
	data = pd.DataFrame(player_stats)
	Total_time_before_rescale = data[0].sum()
	data[0] = data[0]/Total_time_before_rescale*200.0 #rescale the total time for each team in a game. Each game is 40min long, so 40*5=200 min for all players in total
	team_stats = np.array(data.iloc[:,2:].sum())/Total_time_before_rescale*200.0
	return team_stats

def new_players_data_gen(data):
	'''
	To calculate the new player in year i,
	I create a list of all players in yeat i-1,
	and select out all players that are in year i but not in year i-1.
	'''
	all_new_players_data = []
	index=[]
	for year in years:
		this_year_dataframe = pd.read_csv(path+'Players_'+str(year)+'.csv', sep=',')
		this_year_player_names = this_year_dataframe['PlayerName'].tolist()
		last_year_dataframe = pd.read_csv(path+'Players_'+str(year-1)+'.csv', sep=',')
		last_year_player_names = last_year_dataframe['PlayerName'].tolist()
		this_year_new_players = list(set(this_year_player_names)-set(last_year_player_names))
		for new_player in this_year_new_players:
			ID = this_year_dataframe.loc[this_year_dataframe['PlayerName']==new_player]['PlayerID']
			index.append(ID.tolist()[0]-600000-1)
	all_new_players_data = data.iloc[index]
	return all_new_players_data

def main():
	output_features = ['sample_ID',
					'a_miss2_lay', 'a_reb_off', 'a_made2_jump',
					'a_miss2_jump', 'a_assist', 'a_made3_jump',
					'a_block', 'a_reb_def', 'a_foul_pers',
					'a_miss1_free', 'a_made1_free', 'a_miss3_jump',
					'a_turnover', 'a_steal', 'a_made2_dunk',
					'a_made2_lay', 'a_reb_dead', 'a_made2_tip',
					'a_miss2_dunk', 'a_miss2_tip', 'a_foul_tech',
					'b_miss2_lay', 'b_reb_off', 'b_made2_jump',
					'b_miss2_jump', 'b_assist', 'b_made3_jump',
					'b_block', 'b_reb_def', 'b_foul_pers',
					'b_miss1_free', 'b_made1_free', 'b_miss3_jump',
					'b_turnover', 'b_steal', 'b_made2_dunk',
					'b_made2_lay', 'b_reb_dead', 'b_made2_tip',
					'b_miss2_dunk', 'b_miss2_tip', 'b_foul_tech',
					'win']
	sample_ID = 0
	past_player_names = []
	[data, teams] = get_data()
	new_players_data = new_players_data_gen(data)
	new_player_stats = new_players_data.iloc[:,2:-1].mean().tolist()
	if not os.path.isfile(path+'pre_game_teams.csv'):
		with open(path+'pre_game_teams.csv', 'w') as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(output_features)
			for year in years:
				print('Training games in %d... The input data are taken from players stats in %d.' % (year, year - 1))

				players_last_year = pd.read_csv(path+'Players_'+str(year-1)+'.csv', sep=',') #打开上一年的player的信息
				players_this_year = pd.read_csv(path+'Players_'+str(year)+'.csv', sep=',') #打开今年player的信息
				this_year_player_IDs = players_this_year['PlayerID'].tolist() #每个ele的type是int
				this_year_player_names = players_this_year['PlayerName'].tolist()
				last_year_player_IDs = players_last_year['PlayerID'].tolist() #每个ele的type是int
				last_year_player_names = players_last_year['PlayerName'].tolist()

				past_player_names = past_player_names + list(set(last_year_player_names)-set(past_player_names))

				#Calculating team members in each game in this year
				events = list(csv.reader(open(path+'Events_' + str(year) + '.csv', 'r')))
				prev_row = events[1]
				partition_index = []
				for i in range(1,len(events)):
					cur_row = events[i]
					if cur_row[3] != prev_row[3] or cur_row[4] != prev_row[4]:
						prev_row = cur_row
						partition_index.append(i-1)
				print('There are', len(partition_index), 'games in year', year)
				partition_index.insert(0, 0)

				for j in range(len(partition_index) - 1):
					game_j = events[partition_index[j] + 1 : partition_index[j + 1] + 1] #a subset of events corresponding to one game
					W_ID = game_j[0][3]
					L_ID = game_j[0][4]
					W_players_ID = [] #each ele type should be int
					L_players_ID = []
					for k in range(len(game_j)):
						if game_j[k][-3] == W_ID and int(game_j[k][-2]) not in teams and int(game_j[k][-2]) not in W_players_ID:
								W_players_ID.append(int(game_j[k][-2]))
						elif game_j[k][-3] == L_ID and int(game_j[k][-2]) not in teams and int(game_j[k][-2]) not in L_players_ID:
							L_players_ID.append(int(game_j[k][-2]))

					#Calculating WTeam stats
					W_player_stats = player_stats_gen(W_players_ID, this_year_player_names, this_year_player_IDs, last_year_player_names, last_year_player_IDs, data, new_player_stats)
					W_team_stats = team_stats_gen(W_player_stats)

					#Calculating LTeam stats
					L_player_stats = player_stats_gen(L_players_ID, this_year_player_names, this_year_player_IDs, last_year_player_names, last_year_player_IDs, data, new_player_stats)
					L_team_stats = team_stats_gen(L_player_stats)

					#Writing to features.csv
					if sample_ID % 2 == 0:
						row = [sample_ID] + list(W_team_stats) + list(L_team_stats) + [1]
					else:
						row = [sample_ID] + list(L_team_stats) + list(W_team_stats) + [0]
					writer.writerow(row)
					sample_ID += 1

main()
