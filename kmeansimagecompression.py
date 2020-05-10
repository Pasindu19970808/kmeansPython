import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(r'C:\Users\ASUS\Desktop\Python code\Data Science\parrot.jpg')
rows,columns,dim = img.shape
#reshaping the matrix
combined = np.reshape(img,(rows*columns,3), order = 'F')
#normalizing the matrix
combined = combined/255
#combining with the index matrix 
ind = np.zeros((rows*columns,1))
combinedwithindex = np.concatenate((combined,ind),axis = 1)
totalrows = rows*columns
#number of clusters
K = 16
numiter = 10


#random rows to extract for initial centroids
positions = np.random.permutation(rows*columns).tolist()[:K]
#initial centroids
initcentroids = combinedwithindex[np.ix_(positions,(0,1,2))]
#for above operation we can also use initcentroids = combinedwithindex[:,[0,1,2]][positions,:]

#data = {'x1':[3,6,8],'x2':[3,2,5]}
#initcentroids = pd.DataFrame(data = data, columns = ['x1','x2'])
        
for i in range(0,numiter):
    print('Running {} iteration'.format(i))
    for j in range(0,totalrows):
        #list to hold the root mean squares
        rms = [0]*K
        for h in range(0,K):
            #distance between point under consideration and centroid
            dist = combined[j,:] - initcentroids[h,:]
            dist = sum(dist**2)
            rms[h] = dist
        minpos = rms.index(min(rms))
        combinedwithindex[j,3] = minpos
    prevcentroid = initcentroids
    #resetting the initial centroids
    for k in range(0,K):
        #row positions of all rows assigned to K cluster
        indpositions = np.where(combinedwithindex[:,3] == k)
        #the rows 
        closeproxim = combinedwithindex[indpositions][:,[0,1,2]]
        #calculating the mean
        newcentroid = np.mean(closeproxim,axis = 0)
        initcentroids[k,:] = newcentroid
        
finalidx = combinedwithindex[:,3]
#assigning the original "combined" pixels to pixels_recovered array
pixels_recovered = combined
for k in range(0,K):
    finalindpositions = np.where(finalidx == k)
    pixels_recovered[finalindpositions,:] = initcentroids[k,:]

#reshaping into RGB format to have 3 dimensional array(this is the compressed image)
pixelsfordisplay = np.reshape(pixels_recovered,(rows,columns,dim), order = 'F')  
#reshaping into RGB format(original image)
combinedfordisplay = np.reshape(combined,(rows,columns,dim),order = 'F')

cv2.imshow('image0',pixelsfordisplay)
img = img/255
cv2.imshow('image1',img)




    
    
        
        
            
        
            
            
            
            
            
