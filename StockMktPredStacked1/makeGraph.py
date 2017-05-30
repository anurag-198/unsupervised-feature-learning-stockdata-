import csv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

mypath = "C:\\Users\\Dell\\Desktop\\Project8thSem\\CSV_Files_StockPrice"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [mypath + "\\" + f for f in onlyfiles]

for files in onlyfiles:
    #datafile1 = open(files)
    with open(files) as datafile1 :
        name = files.split('\\')
        print(files)
        split_name = name[len(name) - 1]
        filename = split_name.split('.')

        datafile1reader = csv.reader(datafile1)
        y = []
        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        i = 0
        for row in datafile1reader:
            if(i == 0):
                i = i + 1
                continue
            y.append(float(row[2]))

        nelements = 0
        new_y = []
        i = 1
        for element in y:
            new_y.append(element)
            nelements = nelements + 1
            if(nelements == 14):
                mini = min(new_y)
                maxi = max(new_y)
                for j in range(len(new_y)):
                    new_y[j] = (new_y[j] - mini) / (maxi - mini)
                plt.plot(x,new_y)
                plt.savefig("with_axis/" + filename[0] + "_" + str(i) + ".png")
                plt.axis('off')
                plt.savefig("without_axis/" + filename[0] + "_" + str(i) + ".png")
                plt.close()

                i = i + 1
                nelements = 0
                new_y = []