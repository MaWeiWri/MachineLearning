#!/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014年9月9日

@author: MaWei
'''

import numpy as np
import matplotlib.pyplot as plt

import KMeans

INPUT_FILE = './testSet.txt'
K = 4
SSE = K

def loadDataSet(fileName):
    '''
        读取文件，将其转换为数组
        文件内一行一个事务,feature用\t分隔
    '''
    # 声明结果
    result = []
    # 打开文件                
    fp = open(fileName)
    # 对每一行进行处理
    for line in fp.readlines():
        # 将每一行转换为向量形式
        vector = []
        # 按\t切分
        features = line.strip().split('\t')
        # 将每一个元素转换为float类型            
        for feature in features:
            vector.append(float(feature))
        # 将每个事务放入到result中去
            result.append(vector)
        # 返回结果
    return result


def plotResult(raw_data, index,centroid):
    '''
        展示聚类结果
    '''
    
    k = centroid.shape[0]
    
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = fig.add_axes(rect)
    colors = ['red', 'blue', 'green', 'pink', 'yellow']
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                        'd', 'v', 'h', '>', '<']
        
    m = np.size(raw_data, 0)
    for i in range(m):
        clusterClass = int (index[i])
        color = colors[clusterClass]
        markerStyle = scatterMarkers[clusterClass]
        ax.scatter(raw_data[i][0], raw_data[i][1], c=color, marker=markerStyle)
            
    for i in range(k):
        color = 'yellow'
        markerStyle = 'h'
        ax.scatter(centroid[i][0], centroid[i][1], c=color, marker=markerStyle)
    plt.show()
    plt.hold()


if __name__ == '__main__':
    
    kmeans = KMeans.KMeans(K,SSE)
    
    raw = loadDataSet(INPUT_FILE)
    
    # 读取原始数据
    raw_data = np.array(raw)
     
    cost = np.Inf
     
    for i in range(10):
        currentIndex, currentCentroid = kmeans.clusting(raw_data)
        currentCost = kmeans.calculateCost(raw_data, currentIndex, currentCentroid)
        if currentCost < cost :
            cost = currentCost
            index = currentIndex
            centroid = currentCentroid
        print ('Current cost is :' + str(currentCost))
        print ('The best cost is :' + str(cost))
    plotResult(raw_data, index,centroid)
