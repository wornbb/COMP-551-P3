import numpy as np
from  sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage import measure
import cv2
x = pickle.load(open('train_x','rb'))
i = 1

# clean all noise, almost guarantee noise removal, but also damage the characters
x[x<255] = 0
for pics in x:
    pics = cv2.resize(pics,(0,0),fx=4,fy=4)
    # blur the image, try to cure the damage
    blur = cv2.GaussianBlur(pics,(5,5),0.5)
    # sharp it again, prepare for segmentation
    blobs = blur > 30
    blobs_labels = measure.label(blobs, background=0)
    plt.imshow(blobs_labels)
    #plt.imshow(blur)
    plt.savefig("./pic_dump/" + str(i) + ".jpg")
    i += 1
    print(i)
#plt.imshow(blobs_labels, cmap='gray')
#plt.imshow(blobs_labels, cmap='gray')
plt.imshow(x[4])
plt.show()