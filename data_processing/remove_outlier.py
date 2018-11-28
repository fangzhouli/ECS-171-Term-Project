import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

def remove_outlier(root):
	filepath = root + 'post_game_team_diff.csv'
	try:
		df = pd.read_csv(filepath)
	except:
		print("can't open the file")
		exit(0)
	rng = np.random.RandomState(42)
	clf = IsolationForest(behaviour='new', max_samples=100,
	                      random_state=rng, contamination='auto')
	dectec = clf.fit_predict(df)
	outlier = np.where(dectec==-1)[0]
	dropped_csv = df.drop(outlier,axis=0)
	dropped_csv.to_csv(root + 'post_game_team_diff_removed_outlier.csv')


remove_outlier('/Users/jixingwei/Desktop/ECS-171-Group-Project/data_processing/output/')