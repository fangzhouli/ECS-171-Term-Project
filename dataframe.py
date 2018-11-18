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

	
def process_data():
	"""
	This function temporarily process data and get the year, PlayerID and n_match
	returns a dataframe with PlayerID,n_match,year as column

	Note that this function takes a long time to run. It needs further optimization.
	"""
	df = read_data()
	year = set(df['Season'])
	df_g = df.set_index(['Season','DayNum','EventPlayerID'])
	df2 = pd.DataFrame(columns=['PlayerID','year','n_match'])
	for y in year:   
	    c = Counter()
	    for d in set(df_g.loc[(y,),].index.get_level_values(0)):
	        c = c + Counter(set(df_g.loc[(y,d),].index))
	    df_temp = pd.DataFrame.from_dict(c,orient='index',columns=['n_match'])
	    df_temp.index.name = 'PlayerID'
	    df_temp = df_temp.reset_index(level = 0)
	    df_temp['year'] = y
	    df2 = df2.append(df_temp,ignore_index=True)
	return df2
