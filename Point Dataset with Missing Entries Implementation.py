# Licensed under CC-BY 4.0

import pandas as pd

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import math
import random

#DeLi Algorithm

class DeLi:
    def __init__(self, alpha_l, c):
        self.alpha_l = alpha_l
        self.c = c

    def ifPoint(self,A):
        for i in range(len(A)):
           if math.isnan(A[i]):
              return i
        return (-1)

    def point_line(self,point, segment_start, segment_end):
   
        point = np.array(point)
        segment_start = np.array(segment_start)
        segment_end = np.array(segment_end)
    
    # Vector from start to end of the segment
        segment_vector = segment_end - segment_start
    
    # Vector from start of the segment to the point
        point_vector = point - segment_start
    
    # Project point_vector onto segment_vector
        segment_length_squared = np.dot(segment_vector, segment_vector)
        if segment_length_squared == 0:
        # Segment start and end are the same point
            return np.linalg.norm(point_vector)
    
        projection = np.dot(point_vector, segment_vector) / segment_length_squared
        projection = np.clip(projection, 0, 1)
    
    # Find the closest point on the segment
        closest_point = segment_start + projection * segment_vector
    
    # Return the distance from the point to the closest point on the segment
        return np.linalg.norm(point - closest_point)


    


    def line_line(self,A1, A2, B1, B2):

        def closest_point_on_segment(P, Q1, Q2):
       
            Q1 = np.array(Q1)
            Q2 = np.array(Q2)
            P = np.array(P)
            segment_vector = Q2 - Q1
            point_vector = P - Q1
            segment_length_squared = np.dot(segment_vector, segment_vector)
            if segment_length_squared == 0:
               return Q1
        
            projection = np.dot(point_vector, segment_vector) / segment_length_squared
            projection = np.clip(projection, 0, 1)
            return Q1 + projection * segment_vector

        A1 = np.array(A1)
        A2 = np.array(A2)
        B1 = np.array(B1)
        B2 = np.array(B2)
    
    # Calculate distances from segment endpoints to the other segment
        distances = [
         np.linalg.norm(closest_point_on_segment(A1, B1, B2) - A1),
         np.linalg.norm(closest_point_on_segment(A2, B1, B2) - A2),
         np.linalg.norm(closest_point_on_segment(B1, A1, A2) - B1),
         np.linalg.norm(closest_point_on_segment(B2, A1, A2) - B2)
        ]
    
        return min(distances)
        

        
        

    def neighbour(self,A,B):
         a = self.ifPoint(A)
         b = self.ifPoint(B)
         A1, A2 =np.array(A), np.array(A)
         B1, B2 = np.array(B), np.array(B)
         if (a==-1) and (b==-1):
             return np.linalg.norm(A1 - B1) < self.alpha_l
         if a==-1:
             B1[b]=-4
             B2[b]=4
             return  self.point_line(A,B1,B2)< self.alpha_l
         if b==-1:
             A1[b]=-4
             A2[b]=4
             return self.point_line(B,A1,A2) < self.alpha_l
         A1[a]=-4
         A2[a]=4
         B1[b]=-4
         B2[b]=4
         return  self.line_line(A1,A2,B1,B2)< self.alpha_l
             
             
         
         
        
    

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

#importing dataset Sporulation
 

df = pd.read_excel("Sporulation (With Missing Entries).xlsx")
t0 = list(df["t0"])
t0_5 = list(df["t0.5"])
t2 = list(df["t2"])
t5 = list(df["t5"])
t7 = list(df["t7"])
t9 = list(df["t9"])
t11_5 = list(df["t11.5"])

l = [[t0[i],t0_5[i],t2[i],t5[i],t7[i],t9[i],t11_5[i]] for i in range(len(t0))]








# Applying DeLi


DeLi = DeLi(alpha_l=0.6, c=7)
labels = DeLi.fit_predict(l)
print(labels)
clusters = DeLi.clusters(l)
print(clusters)
#first cluster is the set of all outliers







