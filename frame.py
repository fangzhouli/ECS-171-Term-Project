'''
Frame play-by-play row data files.

Usage:
	Change path1 to be the path of your event file, and path2 to be that of players file.
'''

'''
TODO:
	1. Figure out how to deal with the situation when the game is end, but the players
		are still in the arrays, playerID and start
	2. Count the event type other than subin subout.
'''

path1 = '/Users/fzli/Desktop/ECS/171/Project/' + \
		'Basketball_data/PlayByPlay_2010/Events_2010.csv'
path2 = '/Users/fzli/Desktop/ECS/171/Project/' + \
		'Basketball_data/PlayByPlay_2010/Players_2010.csv'

header = ['player_ID', 'season', 'time', 'n_match', 'miss2_lay', 'reb_off', \
			'made2_jump', 'miss2_jump', 'assist', 'made3_jump', 'block'\
			'reb_def', 'foul_pers', 'miss1_free', 'made1_free', 'miss3_jump'\
			'turnover', 'sub_out', 'steal', 'made2_dunk', 'timeout_tv', \
			'made2_lay', 'timeout', 'reb_dead', 'made2_tip', 'miss2_dunk', \
			'miss2_tip', 'foul_tech']

from readFile import readFile

def makeDataCsv():
	events = readFile(path1)
	players = readFile(path2)
	# Initialize empty cells with the first column is player ID
	lines = []
	for elm in players[1:]:
		line = [elm[0], '2010']
		line.extend([0] * 26)
	# Store player ID and initial time of who are playing on the field
	playerID = []
	start = []
	for event in events:
		# If event type is sub_in, record
		if event[10] == 'sub_in':
			palyerID.append(event[9])
			start.append(int(event[7]))
		# If event type is sub_out, check if the player has been recorded
		elif event[10] == 'sub_out':
			# Player is recorded, time is the interval
			if event[9] in playerID:
				old = lines[int(event[9]) - 600001][2]
				new = event[7] - start.pop(playerID.index(event[9]))
				lines[int(event[9]) - 600001][2] = old + new
			# Player is not recorded, time is from the beginning
			else:
				lines[int(event[9]) - 600001][2] = event[7]
			playerID.pop(playerID.index(event[9]))
		# elif event[10] == 'timeout':
		# 	while playerID:
		# 		old = lines[int(playerID[0]) - 600001][2]
		# 		new = event[7] - start[0]
		# 		lines[int(playerID[0]) - 600001][2] = old + new
		# 		playerID.pop(0)
		# 		start.pop(0)
		# else:
		# 	lines[int()]

	# with open

def main():
	makeDataCsv()

if __name__ == '__main__':
	main()
