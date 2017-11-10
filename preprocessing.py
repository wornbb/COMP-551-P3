import csv
import pickle
r = csv.reader(open('.\\Data\\train_y.csv')) # Here your csv file
lines = [l for l in r]

print(len(lines))
for i in range(len(lines)):
    if lines[i] == ['20']:
        lines[i] = ['19']
    if lines[i] == ['21']:
        lines[i] = ['20']
    if lines[i] == ['24']:
        lines[i] = ['21']
    if lines[i] == ['25']:
        lines[i] = ['22']
    if lines[i] == ['27']:
        lines[i] = ['23']
    if lines[i] == ['28']:
        lines[i] = ['24']
    if lines[i] == ['30']:
        lines[i] = ['25']
    if lines[i] == ['32']:
        lines[i] = ['26']
    if lines[i] == ['35']:
        lines[i] = ['27']
    if lines[i] == ['36']:
        lines[i] = ['28']
    if lines[i] == ['40']:
        lines[i] = ['29']
    if lines[i] == ['42']:
        lines[i] = ['30']
    if lines[i] == ['45']:
        lines[i] = ['31']
    if lines[i] == ['48']:
        lines[i] = ['32']
    if lines[i] == ['49']:
        lines[i] = ['33']
    if lines[i] == ['54']:
        lines[i] = ['34']
    if lines[i] == ['56']:
        lines[i] = ['35']
    if lines[i] == ['63']:
        lines[i] = ['36']
    if lines[i] == ['64']:
        lines[i] = ['37']
    if lines[i] == ['72']:
        lines[i] = ['38']
    if lines[i] == ['81']:
        lines[i] = ['39']

writer = csv.writer(open('train_y_preprocessed.csv', 'w', newline=''))
writer.writerows(lines)
pickle.dump(lines,open('train_y_preprocessed','wb'))