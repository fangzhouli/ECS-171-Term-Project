'''
Frame play-by-play row data files.

Usage:
	Change path1 to be the path of your event file, and path2 to be that of players file.
'''

'''
TODO:
	1. Figure out how to deal with the situation when the game is end, but the players
		are still in the arrays, playerID and start
	2. curr the event type other than subin subout.
'''

path1 = '/Users/fzli/Desktop/ECS/171/Project/' + \
		'Basketball_data/PlayByPlay_2010/Events_2010.csv'
path2 = '/Users/fzli/Desktop/ECS/171/Project/' + \
		'Basketball_data/PlayByPlay_2010/Players_2010.csv'

header = ['player_ID', 'season', 'time', 'n_match', 'miss2_lay', 'reb_off', \
			'made2_jump', 'miss2_jump', 'assist', 'made3_jump', 'block', \
			'reb_def', 'foul_pers', 'miss1_free', 'made1_free', 'miss3_jump', \
			'turnover', 'sub_out', 'steal', 'made2_dunk', 'timeout_tv', \
			'made2_lay', 'timeout', 'reb_dead', 'made2_tip', 'miss2_dunk', \
			'miss2_tip', 'foul_tech']

from readFile import readFile

def makeDataCsv():
	curr = 1
	process = 0

	print("Start reading CSV files...")
	events = readFile(path1)
	players = readFile(path2)
	# Initialize empty cells with the first column is player ID
	lines = []
	for elm in players[1:]:
		line = [elm[0], '2010']
		line.extend([0] * 26)
		lines.append(line)
	# Store player ID and initial time of who are playing on the field
	print("Start processing event file...")
	playerID = []
	start = []
	for event in events[1:]:
		# Print complete ratio of the process
		if not round(curr / 25000) == process:
			process = round(curr / 25000)
			print('Process complete: ', process, '%')
		# If event type is sub_in, record
		row = int(event[9]) - 600001
		if event[10] == 'sub_in':
			playerID.append(event[9])
			start.append(int(event[7]))
		# If event type is sub_out, check if the player has been recorded
		elif event[10] == 'sub_out':
			# Player is recorded, time is the interval
			if event[9] in playerID:
				lines[row][2] += int(event[7]) - start.pop(playerID.index(event[9]))
				playerID.pop(playerID.index(event[9]))
			# Player is not recorded, time is from the beginning
			else:
				lines[row][2] += int(event[7])
		else:
			column = header.index(event[10])
			lines[row][column] += 1
		# Check if it is the last event for a match. Empty playerID and start if it is
		if curr + 1 == len(events) or events[curr + 1][7] < event[7]:
			while playerID:
				lines[row][2] += int(event[7]) - start[0]
				playerID.pop(0)
				start.pop(0)
		curr += 1
		
	print(lines)
	# with open

def main():
	makeDataCsv()

if __name__ == '__main__':
	main()
