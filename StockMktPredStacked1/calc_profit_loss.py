import csv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

mypath = "C:\\Users\\Dell\\Desktop\\Project8thSem\\CSV_Files_StockPrice_Large"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [mypath + "\\" + f for f in onlyfiles]

lis = []
for files in onlyfiles:
    with open(files) as datafile1:
        name = files.split('\\')
        split_name = name[len(name) - 1]
        filename = split_name.split('.')

        datafile1reader = csv.reader(datafile1)
        openval = []
        closeval = []
        profitval = []
        i = 0
        for row in datafile1reader:
            if(i == 0):
                i = i + 1
                continue
            openval.append(float(row[1]))
            closeval.append(float(row[4]))

        nelements = 0
        p = 0
        for i in range(len(openval)):
            p = p + (closeval[i] - openval[i])
            nelements = nelements + 1
            if(nelements == 14):
                profitval.append(p)
                nelements = 0
                p = 0

        lis.append([filename[0]] + profitval)

print(lis)

with open("profit_large.csv", 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(lis)

