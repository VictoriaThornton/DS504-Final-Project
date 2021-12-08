import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward

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


# print(dataset.columns)
# print(dataset.head)

#create & show dendrogram
#truncate data to 20,000 points due to data limit for sklearn forward linkage function
linkages = ward(dataset.truncate(before = None, after=20000))
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Cluster Distance')
dendrogram(linkages)
plt.show()

#create & show scatter plot clusters
dataset = dataset.truncate(before = None, after=20000)
agg = AgglomerativeClustering(n_clusters=15)
assignment = agg.fit_predict(dataset)
plt.scatter(dataset.iloc[:,2], dataset.iloc[:,3], c = assignment)
plt.title('Time, Vote')
plt.show()
