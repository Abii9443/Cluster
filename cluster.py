#!/usr/bin/env python
# coding: utf-8

# # K Means Clustering

# In[178]:


#Import Dependencies
import numpy as np # numpy array
import pandas as pd # for dataframes
from sklearn.cluster import KMeans #cluster Algorithm
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import pickle

# Data Collection and Analysis

# In[179]:


df=pd.read_csv('Mall_Customers.csv') #load data from CSV file
df.head()


# In[180]:


df.isna().sum() #check missing value


# In[181]:


df.info()


# In[182]:


df.shape


# In[183]:


#df['Gender']=df['Gender'].map({'Male':1 ,'Female':0})
print(df.head())


# In[184]:



# In[185]:


from sklearn.preprocessing import StandardScaler

x=df.iloc[:, [3,4]].values
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x.columns= ['Annual Income (k$)','Spending Score (1-100)']
x


# In[186]:




# In[187]:


#choose number of clusters by using WCSS->with in cluster sum of sqre
wcss=[]
for i in range(1,16):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=36)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)  # gives wcss value for each clusters
print(wcss)


# In[188]:


#elbow graph



# In[189]:


for i in range(2,16):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(x).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(x,labels,metric="euclidean",sample_size=1000,random_state=200)))


# In[190]:


kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++",random_state=36)
Y=kmeans.fit_predict(x)
x['cluster'] = Y
x.tail()


# In[191]:


silhouette_score(x,Y)


# In[192]:





# 
# # Agglomerative Clustering

# In[193]:


from sklearn.cluster import AgglomerativeClustering



# In[194]:


agglom = AgglomerativeClustering(n_clusters = 5,affinity='euclidean', linkage = 'complete')


# In[195]:





# In[196]:





# In[197]:


agglo_pred=agglom.fit_predict(x)
agglo_pred


# In[198]:


silhouette_score(x,agglo_pred)


# In[199]:




# # DBSCAN

# In[200]:


from sklearn.cluster import DBSCAN


# In[201]:





# In[232]:


dbscan=DBSCAN(eps=11,min_samples=6)


# In[233]:


labeldb=dbscan.fit_predict(x)
labeldb


# In[234]:


#print(silhouette_score(x,labeldb))


# In[235]:


print(np.unique(labeldb))


# In[236]:

pickle.dump(dbscan,open("dbscan.pkl","wb"))
pickle.dump(agglom,open("agglo.pkl","wb"))
pickle.dump(kmeans,open("KMeans.pkl","wb"))




# In[ ]:





# In[ ]:




