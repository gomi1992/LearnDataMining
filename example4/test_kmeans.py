import numpy as np
import time
import matplotlib.pyplot as plt
import kmeans

## step 1: load data
print("step 1: load data...")
dataSet = []
fileIn = open('testSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...
print("step 2: clustering...")
dataSet = np.mat(dataSet)
k = 4
centroids, clusterAssment = kmeans.kmeans(dataSet, k)

## step 3: show the result
print("step 3: show the result...")
kmeans.showCluster(dataSet, k, centroids, clusterAssment)