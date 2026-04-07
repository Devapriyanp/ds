import os
import cv2
import numpy as newnp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

newdata = []
newlabels = []

newpath = "dataset"  


for newfolder in os.listdir(newpath):
    newfolder_path = os.path.join(newpath, newfolder)
    
    for newimg in os.listdir(newfolder_path):
        newimg_path = os.path.join(newfolder_path, newimg)
        
        newimage = cv2.imread(newimg_path)
        newimage = cv2.resize(newimage, (50, 50))
        newimage = newimage.flatten()
        
        newdata.append(newimage)
        newlabels.append(newfolder)

newX = newnp.array(newdata)
newy = newnp.array(newlabels)


newX_train, newX_test, newy_train, newy_test = train_test_split(newX, newy, test_size=0.3)


newmodel = KNeighborsClassifier(n_neighbors=3)
newmodel.fit(newX_train, newy_train)

print("Accuracy:", newmodel.score(newX_test, newy_test))