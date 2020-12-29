#Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.
#Data Description:
#Murder -- Muder rates in different places of United States
#Assualt- Assualt rate in different places of United States
#UrbanPop - urban population in different places of United States
#Rape - Rape rate in different places of United States
import pandas as pd
import matplotlib.pylab as plt 

crime=pd.read_csv(r"filepath\crime_data.csv")

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
#gapminder.rename(columns={'lifeExp':'life_exp'}, index={0:'zero',1:'one'}, inplace=True)
crime.rename(columns={"Unnamed 0:":"USAStates"},inplace=True)
      
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=6,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.clust=crime.clust.map({0:1,1:2,2:3,3:4,4:5,5:6})
crime=crime.sort_values("clust")
i=list(range(1,51))
crime.index=i

# getting aggregate mean of each cluster
crime.iloc[:,2:].groupby(crime.clust).median()

# creating a csv file 
crime.to_csv(r"filepath\crimehclust.csv",encoding="utf-8")

#Inference
#The Groups are made from more Intense Crime rate upto least Intense Crime Rates but order is different 
#Order Being 4th<1st<6th<5th<3rd<2nd