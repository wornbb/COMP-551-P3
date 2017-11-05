import csv

r = csv.reader(open('predictions_cnn.csv')) # Here your csv file
lines = [l for l in r]

print(len(lines))
for i in range(len(lines)):
    if lines[i][1] == '19':
        lines[i][1] = '20'
    if lines[i][1] == '20':
        lines[i][1] = '21'
    if lines[i][1] == '21':
        lines[i][1] = '24'
    if lines[i][1] == '22':
        lines[i][1] = '25'
    if lines[i][1] == '23':
        lines[i][1] = '27'
    if lines[i][1] == '24':
        lines[i][1] = '28'
    if lines[i][1] == '25':
        lines[i][1] = '30'
    if lines[i][1] == '26':
        lines[i][1] = '32'
    if lines[i][1] == '27':
        lines[i][1] = '35'
    if lines[i][1] == '28':
        lines[i][1] = '36'
    if lines[i][1] == '29':
        lines[i][1] = '40'
    if lines[i][1] == '30':
        lines[i][1] = '42'
    if lines[i][1] == '31':
        lines[i][1] = '45'
    if lines[i][1] == '32':
        lines[i][1] = '48'
    if lines[i][1] == '33':
        lines[i][1] = '49'
    if lines[i][1] == '34':
        lines[i][1] = '54'
    if lines[i][1] == '35':
        lines[i][1] = '56'
    if lines[i][1] == '36':
        lines[i][1] = '63'
    if lines[i][1] == '37':
        lines[i][1] = '64'
    if lines[i][1] == '38':
        lines[i][1] = '72'
    if lines[i][1] == '39':
        lines[i][1] = '81'

writer = csv.writer(open('predictions_cnn_postprocessed.csv', 'w', newline=''))
writer.writerows(lines)