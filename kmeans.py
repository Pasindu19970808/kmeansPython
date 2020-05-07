import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\ASUS\Desktop\Python code\Data Science\kmeansdata.csv')
dimensions = len(df.columns)
#number of clusters
K = 3
numiter = 10
#amount of rows in df
rownum = len(df)
#attaching index column to df
df['Cluster Index'] = [0]*rownum
newdimension = len(df.columns)



#random rows to extract for initial centroids
positions = np.random.permutation(rownum).tolist()[:K]
#initial centroids
initcentroids = df.iloc[positions,list(range(dimensions))]
initcentroids = initcentroids.reset_index(drop=True)
        
for i in range(0,numiter):
    for j in range(0,rownum):
        #list to hold the root mean squares
        rms = [0]*K
        for h in range(0,K):
            #distance between point under consideration and centroid
            dist = df.iloc[j,list(range(0,dimensions))] - initcentroids.iloc[h,list(range(0,dimensions))]
            dist = (sum(dist))**2
            rms[h] = dist
        minpos = rms.index(min(rms))
        df.iloc[j,-1] = minpos
    prevcentroid = initcentroids
    #resetting the initial centroids
    for k in range(0,K):
        initcentroids.iloc[k,:] = df.loc[lambda df:df['Cluster Index'] == k].iloc[:,[0,1]].mean(axis = 0)

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmapcols = plt.get_cmap('gist_rainbow')
colnum = np.linspace(0,1,len(range(0,K)))
colors = iter(cmapcols(colnum))
for k in range(0,K):
    x = df.loc[lambda df:df['Cluster Index'] == k].iloc[:,0]
    y = df.loc[lambda df:df['Cluster Index'] == k].iloc[:,1]
    ax1.scatter(x,y,marker = '*',label = k,color = next(colors))
    
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.legend()
    
    
    
        
        
            
        
            
            
            
            
            
