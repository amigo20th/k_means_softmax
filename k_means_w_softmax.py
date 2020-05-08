from scipy.special import softmax
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

df = pd.read_csv("cc_general.csv")
cols = []
for col in df.columns:
    cols.append(col.lower())
df = df.dropna()
df.columns = cols
cust_id_col = df.cust_id
df = df.drop(['cust_id'], axis=1)

df_norm = (df - df.min()) / (df.max() - df.min())

# KMeans determinate k
distortions = []
K = range(1, 5)
for k in K:
    kmeansModel = KMeans(n_clusters=k).fit(df_norm)
    distortions.append(sum(np.min(cdist(df_norm, kmeansModel.cluster_centers_, 'euclidean'), axis= 1)) / df_norm.shape[0])

plt.plot(K, distortions, 'bx-')
plt.show()

# We select k = 3

kmeansModel = KMeans(n_clusters=3).fit(df_norm)
#print(kmeansModel.cluster_centers_)
# distances from centers of the cluster and all points in the DataFrame
dist_list = cdist(df_norm, kmeansModel.cluster_centers_, 'euclidean')
# Apply Softmax in the distances
out_softmax = []
for ind in range(len(dist_list)):
    out_softmax.append(softmax(dist_list[ind]))

#the probabilities of the fuzzy KMeans are:
print(out_softmax)