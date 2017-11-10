import numpy as np
import pickle

np.random.seed(123)

mapOutputs = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 20:19, 21:20, 24:21, 25:22, 27:23, 28:24, 30:25, 32:26, 35:27, 36:28, 40:29, 42:30, 45:31, 48:32, 49:33, 54:34, 56:35, 63:36, 64:37, 72:38, 81:39}
unmapOutputs = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:20, 20:21, 21:24, 22:25, 23:27, 24:28, 25:30, 26:32, 27:35, 28:36, 29:40, 30:42, 31:45, 32:48, 33:49, 34:54, 35:56, 36:63, 37:64, 38:72, 39:81}


def sigmoid(inp):
    return 1/(1+np.exp(-inp))

def sigmoid_deriv_with_output(oup):
    return oup*(1-oup)

alpha = 100
hiddenSize = 32
batch_size = 128


x = pickle.load(open('trainX.p', 'rb'))    # load from text
x /= 255

y = pickle.load(open('trainY.p', 'rb'))
for entry in range(0, len(y)):
    y[entry] = mapOutputs[y[entry]]
yArr = y.reshape(len(y),1)
yArr

yCategorized = []
for idx in range(0, len(yArr)):
    tmpArr = np.zeros((1,40))
    tmpArr[0][int(yArr[idx][0])] = 1
    yCategorized.append(np.ndarray.tolist(tmpArr))
yCategorized = np.asarray(yCategorized)



xAnd = np.array([[1,0],[0,1],[0,0],[1,1]])
yAnd = np.array([[0],[0],[0],[1]])

xVal = x[45000:]
yVal = yCategorized[45000:]

w_0 = np.random.random((4096,hiddenSize))
w_1 = np.random.random((hiddenSize,40))

for counter in range(0, 25):
    
    for whichX in range(0, 300-batch_size, batch_size):
        print(whichX)
        inputLayer = x[whichX:whichX+batch_size]
        hiddenLayer_1 = sigmoid(np.dot(inputLayer, w_0))
        outputLayer = sigmoid(np.dot(hiddenLayer_1, w_1))

        outputSquaredError = []
        for idx2 in range(0, len(outputLayer)):
            outputSquaredError.append(np.ndarray.tolist((yCategorized[idx2] - outputLayer[idx2])))
        outputSquaredError = np.asarray(outputSquaredError)
        
        outputDeriv = sigmoid_deriv_with_output(outputLayer)
        outputDelta = outputSquaredError*outputDeriv
        hiddenLayer_1_Error = outputDelta.dot(w_1.T)
        hiddenLayer_1_Deriv = sigmoid_deriv_with_output(hiddenLayer_1)
        hiddenLayer_1_Delta = hiddenLayer_1_Error*hiddenLayer_1_Deriv

        w_1_Update = []
        w_0_Update = []
        w_1_Update_Temp = (hiddenLayer_1.T.dot(outputDelta))
        for idx in range(0, len(w_1_Update_Temp)):
            w_1_Update.append(w_1_Update_Temp[idx][0])
        w_1_Update = np.asarray(w_1_Update)
        w_1 = w_1 - alpha*w_1_Update
        w_1_Update_Temp = []
        w_1_Update = []

        w_0_Update_Temp = (inputLayer.T.dot(hiddenLayer_1_Delta))
        for idx in range(0, len(w_0_Update_Temp)):
            w_0_Update.append(w_0_Update_Temp[idx][0])
        w_0_Update = np.asarray(w_0_Update)
        w_0 = w_0 - alpha*w_0_Update
        w_0_Update_Temp = []
        w_0_Update = []
    
    
    
    

    if counter % 2 == 0:
        print('Training Loss:', np.mean(outputSquaredError))
        numCorrect = 0
        totalNum = 0
        for valEntry in range(0, len(xVal)):
            totalNum += 1
            tmpInput = xVal[valEntry]
            h1 = sigmoid(np.dot(tmpInput, w_0))
            tmpOut = sigmoid(np.dot(h1, w_1))
            if np.argmax(tmpOut) == np.argmax(yVal[valEntry]):
                numCorrect += 1
        print('Validation Accuracy:', numCorrect/totalNum)