#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:21:31 2017

@author: Joelson Antonio dos Santos
"""
# Classic kmeans algorithm in python3

# PRINCIPAL REFERENCE: MacQueen, J. B. (1967). Some Methods for classification and Analysis of Multivariate Observations. 
# Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability. 1. 
# University of California Press. pp. 281–297

# Input: kmeans("pathFile", numberOfClusters, numberOfIteration)
# Output: list of cluster labels and final centroids

import numpy as np
import sys

# function used only to read a dataset
def input_dataset(file):
    return np.loadtxt(file, dtype='float')

# caculate euclidean distance between two vectors
def calculate_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2, ord=None)

def kmeans(file, k, iteration):
    # reading d-dimensional dataset
    dataset = input_dataset(file)
    row_dataset = len(dataset)
    clusters = [0] * row_dataset
    # choosing initial seeds
    id_samples = np.random.choice(row_dataset, k, replace=False)
    centroids = np.array(dataset[id_samples,])
    cumulative_centroids = centroids
    old_sum = [0] * k
    new_sum = [-1] * k
    count_members = np.array([0] * k)
    count = 0
    # kmeans iteration
    while count < iteration and old_sum != new_sum:
        new_sum = old_sum
        old_sum = [0] * k
        for j in range(0, row_dataset):
           min_distance = sys.float_info.max
           cluster_id = 0
           for i in range(0, k): 
               distance = calculate_distance(dataset[j,], centroids[i,])
               if min_distance > distance:
                   min_distance = distance
                   cluster_id = i
           old_sum[cluster_id] = old_sum[cluster_id] + min_distance
           clusters[j] = cluster_id
           cumulative_centroids[cluster_id] = cumulative_centroids[cluster_id] + dataset[j]
           count_members[cluster_id] =  count_members[cluster_id] + 1 
        # updating cetroids
        for i in range(0, k):
           if count_members[i] != 0:
               centroids[i] = cumulative_centroids[i] / count_members[i]  
           else:
               # selecting new seed
               centroids[i] = dataset[np.random.choice(row_dataset, 1, replace=False),]
        count_members = [0] * k  
        cumulative_centroids = [0] * k
        count = count + 1
        # return cluster labels and final centroids
    return [clusters, centroids]
