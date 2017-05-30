a = [0,1,1]
b = [2,3,4]
c = zip(a,b)

c = sorted(c, key = lambda x :x[1])
print(c)