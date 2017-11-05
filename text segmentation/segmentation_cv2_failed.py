import numpy as np
import sklearn as sl
import matplotlib.pyplot as plt
import pickle
import cv2
# fast load formatted training data
x = pickle.load(open('clean_x','rb'))
y = pickle.load(open('train_y','rb'))

# Clean the image, only need to run once.
#x[x<255] = 0
#pickle.dump(x,open('clean_x','wb'))
img = x[0]
mser = cv2.MSER_create()
#Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1]*128, img.shape[0]*128))
gray = img
vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
regions = mser.detectRegions(gray.astype(np.uint8))
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
cv2.polylines(vis, hulls, 100, (0,255,0))

cv2.namedWindow('img', 0)
cv2.imshow('img', vis)
while(cv2.waitKey()!=ord('q')):
    continue
cv2.destroyAllWindows()
