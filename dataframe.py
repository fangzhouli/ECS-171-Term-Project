import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import Counter

def read_data():
	"""
	This function reads Events data from 2010-2018 playbyplay/Events.csv
    It will return a pandas dataframe
	"""
	filepath = os.getcwd()
	Allfilenames =  listdir(filepath)
	filenames = [i for i in Allfilenames if 'PlayByPlay' in i]

	df = pd.DataFrame()
	for name in filenames:
		_, year = name.split('_')
		temp = pd.read_csv(name + '/Events_' + year + '.csv')
		df = df.append(temp)
	return df
