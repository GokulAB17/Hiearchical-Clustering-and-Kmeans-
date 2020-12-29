#Perform clustering (Both hierarchical and K means clustering) for the airlines data to obtain optimum number of clusters. 
#Draw the inferences from the clusters obtained.
#Data Description:
#The file EastWestAirlinescontains information on passengers who belong to an airline’s frequent flier program. For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers
#ID --Unique ID
#Balance--Number of miles eligible for award travel
#Qual_mile--Number of miles counted as qualifying for Topflight status
#cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
#cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
#cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:
#1 = under 5,000
#2 = 5,000 - 10,000
#3 = 10,001 - 25,000
#4 = 25,001 - 50,000
#5 = over 50,000
#Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
#Bonus_trans--Number of non-flight bonus transactions in the past 12 months
#Flight_miles_12mo--Number of flight miles in the past 12 months
#Flight_trans_12--Number of flight transactions in the past 12 months
#Days_since_enrolled--Number of days since enrolled in flier program
#Award--whether that person had award flight (free flight) or not
import pandas as pd
import matplotlib.pylab as plt 
import numpy as np
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 


airlines=pd.read_excel(r"filePath/EastWestAirlines.xlsx",sheet_name="data")
airlines.columns
airlines.head()

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines.iloc[:,1:])

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


# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=5,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_complete.labels_)

#Kmeans method
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    

# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)    

# Selecting 6 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=6) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

#Maping clusters from 1 to 5 and sorting records as per clusters for better representation
airlines= airlines.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
airlines.clust=airlines.clust.map({0:1,1:2,2:3,3:4,4:5,5:6})
airlines=airlines.sort_values("clust")
i=list(range(1,4000))
airlines.index=i

#EDA over each clusters
data =airlines.iloc[:,2:12].groupby(airlines.clust).mean()

#Storing new clustered records in your system
airlines.to_csv(r"file path\EastWestAirlines.csv", encoding="UTF-8")

#Inferences from clusters
#Clustering from Hierachical one is insignificant for huge dataset as dendrogram plot is 
#not able to divide such huge records as it is more congested and separation is not achieved 
#Estimation of no of clusters is not accurate by Hiearchical clustering 
#K means provide solution to this problem as we can easily decide number of clusters from scree plot
# For a large no of records (dataset ) K means clustering is significant 
#over Hierarchical clustering.