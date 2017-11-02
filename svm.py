import numpy as np
from sklearn.svm import SVC
import csv
import matplotlib.pyplot as plt

x = np.loadtxt("train_x.csv", delimiter=",")    # load from text
print("train_x is loaded")
y = np.loadtxt("train_y.csv", delimiter=",")
print("train_y is loaded")
z = np.loadtxt("test_x.csv", delimiter=",")
print("test_x is loaded")

clf = SVC()
clf.fit(x, y)   # x[nsample, nfeature]
print("Your SVM is ready")
predictions = clf.predict(z)
print("Writing results into predictions_lr.csv...")

with open('predictions_lr.csv', 'w', newline='') as f:
    fieldnames = ["Id", "Label"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predictions)):
        writer.writerow({'Id': i+1, 'Label': np.uint8(predictions[i])})