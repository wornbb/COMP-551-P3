import numpy as np
from  sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage import measure
import cv2
x = pickle.load(open('clean_x','rb'))
i = 1
for pics in x:
    blur = cv2.blur(pics,(6,6))
    blur[blur>255/8] = 255
    blobs = blur == 255
    blobs_labels = measure.label(blobs, background=0)
    plt.imshow(blobs_labels)

    plt.savefig("./pic_dump/" + str(i) + ".jpg")
    i += 1
#plt.imshow(blobs_labels, cmap='gray')
plt.imshow(blobs_labels, cmap='gray')
plt.show()