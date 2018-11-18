'''
Read csv files and return a list
'''

import csv

root = '/Users/fzli/Desktop/ECS/171/Project/Basketball_data/'
path1 = root + 'PlayByPlay_2010/Events_2010.csv'
path2 = root + 'PlayByPlay_2010/Players_2010.csv'

def readFile(path):
	with open(path, 'r') as file:
		reader = csv.reader(file)
		return list(reader)

def main():
	print(readFile(path2))

if __name__ == '__main__':
	main()
