import numpy as np
import pickle
from copy import deepcopy


xData = pickle.load(open('trainX.p', 'rb'))
yData = pickle.load(open('trainY.p', 'rb'))
xTestData = pickle.load(open('testX.p', 'rb'))


xDataProc = deepcopy(xData)

for img in xDataProc:
    thresh = sum(img)/4096 + 1*np.std(img)
    for idx in range(0, len(img)):
        if img[idx] > thresh:
            img[idx] = 1
        else:
            img[idx] = 0

pickle.dump(xDataProc, open('x01Proc.p', 'wb'))

xTestDataProc = deepcopy(xTestData)

for img in xTestDataProc:
    thresh = sum(img)/4096 + 1*np.std(img)
    for idx in range(0, len(img)):
        if img[idx] > thresh:
            img[idx] = 1
        else:
            img[idx] = 0

pickle.dump(xTestDataProc, open('xTest01Proc.p', 'wb'))