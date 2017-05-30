import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
datafile1 = open("CSV_Files_StockPrice/cyient_ns.csv")
datafile1reader = csv.reader(datafile1)
x = []
y = []
i = 0
for row in datafile1reader:
    """print(row[4])"""
    if(i == 0):
        i = i+1
        continue
    y.append(float(row[4]))
    x.append(row[0])


print(x)
print(y)
"""x.reverse()
y.reverse()"""
newx = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in x]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

plt.plot(newx,y)
plt.show()