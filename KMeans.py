import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filename = 'Video_Games.csv'

X = pd.read_csv(filename, low_memory=False,) 

dataset = X

#data editing to just numerical columns: overall, verified, unixReviewTime, vote
dataset.drop(dataset.columns[[2,3,4,5,6,7,10,11]], axis=1, inplace = True)
dataset = dataset.replace("'",' ', regex=True)
dataset = dataset.replace(",",' ', regex=True)
dataset = dataset.replace(" ",'', regex=True)

#change NaN votes to 0 values
dataset['vote'] = dataset['vote'].fillna(0)


#create k-means clustering model
kmeansModel = KMeans(n_clusters = 5)
kmeansModel.fit(dataset)
plt.scatter(dataset.iloc[:,1], dataset.iloc[:,3], c = kmeansModel.labels_)
plt.title('k-means 1,3: 5')
plt.show()

