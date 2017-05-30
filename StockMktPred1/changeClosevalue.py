import csv
datafile1 = open("CSV_Files_StockPrice/wipro_ns.csv")
datafile1reader = csv.reader(datafile1)
close = []
date = []
i = 0
for row in datafile1reader:
    """print(row[4])"""
    if(i == 0):
        i = i+1
        continue
    close.append(float(row[4]))
    date.append(row[0])
maxi = max(close)
mini = min(close)
for j in range (len(close)):
    close[j] = (close[j] - mini ) / (maxi - mini)

print (close)

with open("wipro_ns.csv", 'w') as csvfile :
    writer = csv.writer(csvfile)
    rows = zip(date, close)

    for row in rows:
        writer.writerow(row)


