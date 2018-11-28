from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import plotnine as p9

def drawTSNE(root):
	filepath = root + 'post_game_team_diff.csv'
	try:
		df = pd.read_csv(filepath)
	except:
		print("can't open the file")
		exit(0)
	X_embedded = TSNE(n_components=2).fit_transform(df)
	df_tsne = pd.DataFrame()
	df_tsne['x'] = X_embedded[:,0]
	df_tsne['y'] = X_embedded[:,1]

	p9.ggplot(df_tsne) \
	+p9.aes('x','y') \
	+p9.geom_point(color='blue',size=0.01) \
	+p9.labels.ggtitle('high dimensional data visualization by t-SNE')

drawTSNE('/Users/jixingwei/Desktop/ECS-171-Group-Project/data_processing/output/')