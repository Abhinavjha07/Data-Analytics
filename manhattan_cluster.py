import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def closestPoint_man(p, centers):
    bestIndex = 0
    closest = float('+inf')
    for i in range(len(centers)):
        tempDist = np.sum(abs(p-centers[i]))
        if tempDist < closest:
            closest = tempDist
            bestIndex = i

    return bestIndex


lines = spark.read.text('/content/data.txt').rdd.map(lambda r: r[0])
data = lines.map(parseVector).cache()
max_iter = 20
convergeDist = 0.001
K = 10
kPoints_near = pd.read_csv('/content/near.txt',sep = ' ').values

tempDist = 1.0
L = []
x = 0
for _ in range(max_iter):
    closest = data.map(lambda p: (closestPoint_man(p, kPoints_near), (p,1)))
    pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
    newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

    tempDist = sum(np.sum(abs(kPoints_near[iK] - p)) for (iK,p) in newPoints)
    L.append([x,tempDist])
    x += 1
    for (iK, p) in newPoints:
        kPoints_near[iK] = p

print('Final centers: '+str(kPoints_near))

style.use('ggplot')
X = np.array(L)
plt.figure(figsize=(12,6))
plt.xlabel("Iteration")
plt.ylabel("Manhattan Distance")
plt.title("NEAR.txt")
plt.plot(X[:,0],X[:,1])
plt.show()


kPoints_far = pd.read_csv('/content/far.txt',sep = ' ').values
tempDist = 1.0
L = []
x = 0
for _ in range(max_iter):
    closest = data.map(lambda p: (closestPoint_man(p, kPoints_far), (p,1)))
    pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
    newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

    tempDist = sum(np.sum(abs(kPoints_far[iK] - p)) for (iK,p) in newPoints)
    L.append([x,tempDist])
    x += 1
    for (iK, p) in newPoints:
        kPoints_far[iK] = p

print('Final centers: '+str(kPoints_far))


style.use('ggplot')
X = np.array(L)
plt.figure(figsize=(12,6))
plt.xlabel("Iteration")
plt.ylabel("Manhattan Distance")
plt.title("FAR.txt")
plt.plot(X[:,0],X[:,1])
plt.show()

