from scipy import special
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
    kmeansModel = KMeans(n_clusters=k).fit(df_norm.values)
    distortions.append(sum(np.min(cdist(df_norm.values, kmeansModel.cluster_centers_, 'euclidean'), axis= 1)) / df_norm.shape[0])

plt.plot(K, distortions, 'bx-')
plt.show()