'''
Frame play-by-play row data files.

Usage:
	Change path1 to be the path of your event file, and path2 to be that of players file.
'''
root = '/../NCAA_data/'

header = ['player_ID', 'season', 'time', 'n_match', 'miss2_lay', 'reb_off', \
			'made2_jump', 'miss2_jump', 'assist', 'made3_jump', 'block', \
			'reb_def', 'foul_pers', 'miss1_free', 'made1_free', 'miss3_jump', \
			'turnover', 'sub_out', 'steal', 'made2_dunk', 'timeout_tv', \
			'made2_lay', 'timeout', 'reb_dead', 'made2_tip', 'miss2_dunk', \
			'miss2_tip', 'foul_tech', 'name']

from readFile import readFile

import os
import csv

def makeDataCsv():
	curr = 0
	process = 0

	print("Start reading CSV files...")
	events = []
	players = []
	for i in range(9):
		year = 2010 + i
		print("File " + str(year) + '...')
		filepath = root + '/Events_' + str(year) + '.csv'
		events.extend(readFile(filepath))
		filepath = root + '/Players_' + str(year) + '.csv'
		players.extend(readFile(filepath))

	offset = int(players[0][0])
	divisor = round(len(events) / 100)
	# Initialize empty cells with the first column is player ID
	lines = []
	for elm in players:
		if elm[0] == '648095':
			for i in range(642767, 648095):
				line = [i]
				line.extend([0] * 28)
				lines.append(line)
		line = [elm[0], elm[1]]
		line.extend([0] * 26)
		line.append(elm[3])
		lines.append(line)
	# Store player ID and initial time of who are playing on the field
	print("Start processing event file...")
	hasPlayed = []
	playerID = []
	start = []
	for event in events:
		# Print complete ratio of the process
		if not round(curr / divisor) == process:
			process = round(curr / divisor)
			print('Process complete: ', process, '%')
		# If event type is sub_in, record
		row = int(event[9]) - offset
		if event[10] == 'sub_in':
			if not event[9] in hasPlayed:
				hasPlayed.append(event[9])
			playerID.append(event[9])
			start.append(int(event[7]))
		# If event type is sub_out, check if the player has been recorded
		elif event[10] == 'sub_out':
			# Player is recorded, time is the interval
			if not event[9] in hasPlayed:
				hasPlayed.append(event[9])
			if event[9] in playerID:
				lines[row][2] += int(event[7]) - start.pop(playerID.index(event[9]))
				playerID.pop(playerID.index(event[9]))
			# Player is not recorded, time is from the beginning
			else:
				lines[row][2] += int(event[7])
		else:
			# The player has been on the field from the beginning
			# if not event[9] in hasPlayed:
			# 	hasPlayed.append(event[9])
			# if not event[9] in playerID:
			# 	playerID.append(event[9])
			# 	start.append(0)
			if event[9] in hasPlayed:
				if not event[9] in playerID:
					playerID.append(event[9])
					start.append(int(event[7]))
			else:
				hasPlayed.append(event[9])
				playerID.append(event[9])
				start.append(0)
			column = header.index(event[10])
			lines[row][column] += 1
		# Check if it is the last event for a match. Empty playerID and start if it is
		if curr + 1 == len(events) or int(event[7]) > int(events[curr + 1][7]):
			while hasPlayed:
				lines[int(hasPlayed[0]) - offset][3] += 1
				hasPlayed.pop(0)
			while playerID:
				lines[int(playerID[0]) - offset][2] += max(2400, int(event[7])) - start[0]
				playerID.pop(0)
				start.pop(0)
		curr += 1

	# Write to a new csv file
	if not os.path.isfile('data.csv'):
		with open('data.csv', 'w') as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(header)
			writer.writerows(lines)

def main():
	makeDataCsv()

if __name__ == '__main__':
	main()
