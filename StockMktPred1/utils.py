import cv2
import csv
import numpy as np
from os import listdir
from os.path import isfile, join

def dataset(location) :
    mypath = location
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = [mypath + "/" + f for f in onlyfiles]

    sortedf = sorted(onlyfiles)
    print(sortedf)

    dat = np.zeros((2500,len(onlyfiles)))
    labels = np.zeros((len(onlyfiles), 1), dtype=np.float)
    i = 0

    for files in sortedf:
        img = cv2.imread(files,0)
        dat[:,i] = img.flatten()
        #print(files)
        name = files.split("/")
        name = name[1]
        name = name.split(".")
        name = name[0]
        name = name.split("_")
        name = str(name[len(name) - 1])
        name = name.replace('#','.')
        no = float(name)
        labels[i][0] = no
        i = i + 1
    dat = dat/255
    print(dat.shape)
    print(labels.shape)

    return dat, labels



#dat, labels = dataset("gray500")
####################################### for labels now ############################################
'''
    labels = np.zeros((180, 1), dtype=np.float)
    filename = "profit.csv"
    lis = []
    with open(filename) as f:
        reader = csv.reader(f)
        for rows in reader:
            if (len(rows) > 1):
                lis.append(rows)
    #print(lis)

    i = 0
    for val in sortedf:
        #print(val)
        name = val.split("/")
        name = name[1]
        name = name.split(".")
        name = name[0]
        name = name.split("_ns_")
        nam = name[0]
        no = int(name[1])
        #print(nam, no)
        nam = nam + "_ns"

        for p in lis:
            if p[0] == nam:
                if (float(p[no]) > 0):
                    labels[i][0] = 1
                else:
                    labels[i][0] = 0

                #print(p[0],nam,p[no], float(p[no]))
                break

        i = i + 1
'''


"""
mysol, label = dataset("rgb")

print("printing the solution data   ")
print (mysol)

print("printing the labels   ")
print(label)

print(mysol.shape)
print(label.shape)


"""