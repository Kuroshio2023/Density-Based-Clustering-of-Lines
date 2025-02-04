# Licensed under CC-BY 4.0

import pandas as pd

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import math
import random

#DBSCAN Algorithm

class DeLi:
    def __init__(self, alpha_l, c):
        self.alpha_l = alpha_l
        self.c = c

    def neighbour(self,A,B):
         x1,y1 = A[0]
         x2,y2 = A[1]
         p1,q1 = B[0]
         p2,q2 = B[1]
         if x1 > x2 :
            x1,x2 = x2,x1
            y1,y2 = y2,y1
         if p1 > p2:
            p1,p2 = p2,p1
            q1,q2 = q2,q1
         l1 = LineString([(x1,y1),(x2,y2)])
         l2 = LineString([(p1,q1),(p2,q2)])
         return (l1.distance(l2) < self.alpha_l)
    

    def _region_query(self,data,point_idx):
        neighbors = []
        for i in range(len(data)):
            if self.neighbour(data[point_idx],data[i]):
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, data,labels, point_idx, cluster_id):
        queue = deque([point_idx])
        while queue:
            current_point_idx = queue.popleft()
            if labels[current_point_idx] == -1:
                labels[current_point_idx] = cluster_id
            neighbors = self._region_query(data,current_point_idx)
            if len(neighbors) >= self.c:
                for neighbor in neighbors:
                    if labels[neighbor] == 0:
                        labels[neighbor] = cluster_id
                        queue.append(neighbor)

    def fit_predict(self, data):
        labels = np.zeros(len(data), dtype=int)  # 0: unvisited, -1: noise, positive integer: cluster id
        current_cluster_id = 0

        for i in range(len(data)):
            if labels[i] != 0:
                continue
            neighbors = self._region_query(data,i)
            if len(neighbors) < self.c:
                labels[i] = -1  # mark as noise
            else:
                current_cluster_id += 1
                self._expand_cluster(data, labels, i, current_cluster_id)

        return labels

    def clusters(self,data):
        labels = self.fit_predict(data)
        k = max(labels)
        clusters = [[j for j in range(len(labels)) if labels[j]==i] for i in range(1,k+1)]
        l = [j for j in range(len(labels)) if labels[j]==-1]
        clusters.insert(0,l)
        return clusters

#importing dataset
#Please replace "Dataset.xlsx" by the appropriate dataset name    

df = pd.read_excel("Dataset.xlsx")
x1 = list(df["x1"])
y1 = list(df["y1"])
x2 = list(df["x2"])
y2 = list(df["y2"])

l = [[[x1[i],y1[i]],[x2[i],y2[i]]] for i in range(len(x1))]








# Applying DeLi


DeLi = DeLi(alpha_l ,c)
labels = DeLi.fit_predict(l)
print(labels)
clusters = DeLi.clusters(l)
#first cluster is set of outliers
print(clusters)

#outliers are black
c=0
for i in clusters:
    if c == 0:
        r, g, b = 0, 0, 0
    else:
        r = random.randint(0, 255)/255
        g = random.randint(0, 255)/255
        b = random.randint(0, 255)/255
    for j in i:
        x1,y1=l[j][0]
        x2,y2 = l[j][1]
        plt.plot([x1,x2],[y1,y2],color=(r, g, b), marker='o', linestyle='-')
    c=c+1

plt.legend()
plt.show()






