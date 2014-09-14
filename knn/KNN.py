# -*- coding:utf-8 -*-
'''
Created on 2014年9月9日

@author: 威
'''

import math
import numpy as np


class KNN(object):
    '''
    KNN算法练习
    '''

    def __init__(self, x, y, k):
        '''
        初始化方法，需要指定训练数据集及每条记录对应的标签k
        x:训练数据集 m*n维矩阵，m为事务数，n为feature的数量
        y:训练数据集对应的标签， 类型为列表，按顺序存放m个值，m为事务数，代表训练数据的每一行属于哪一种类别
        k:knn算法指定的k
        '''
        self.x = x
        self.y = y
        self.k = k
        self._normalizeTrainingSet();
        
        
    def _normalizeTrainingSet(self):
        '''
        对训练数据集的所有维度进行归一化操作
        '''
        # 获取事务总数m
        m = np.shape(self.x)[0]
        # 计算每个feature的均值
        x_mean = np.mean(self.x, 0)
        # 对数据进行归一化
        # 每个事务减去其均值
        x_normal = self.x - np.tile(x_mean, (m, 1))
        # 再除以均值
        x_normal = x_normal / x_mean
        # 赋予新的归一化后的x与各属性均值
        self.x_normal = x_normal
        self.x_mean = x_mean
        
    def _getDistance(self, v1, v2):
        '''
        计算某个向量与训练数据之间的距离(欧氏距离)
        v1: 1*n矩阵
        v2: m*n矩阵
        '''
        # 计算数值差
        temp = v1 - v2
        # 计算平方和
        distance = np.sum(temp * temp) 
        # 开方即得到欧氏距离
        distance = math.sqrt(distance)
        return distance
    
    def _normalize(self, inputData):
        '''
        新数据有分类需求，需要先对其进行归一化操作
        '''
        # 每个feature减去其均值
        x = (inputData - self.x_mean)
        # 再除以均值即可
        x = x / self.x_mean
        return x
    
    
    def _getTopK(self, inputData):
        '''
        传入一个事务(n*1)，计算距离到它最近几个事务的标签
        '''
        # 声明结果数组，用与存放最近几个事务的标签
        result = []
        # 获取事务总数m
        m = np.shape(self.x)[0]
        # 每个原始数据点减去输入数据的feature
        differ = self.x_normal - np.tile(inputData, (m, 1))
        # 平方求和再开方即得到输入事务到每个训练数据点的距离
        differ = differ * differ
        distance = np.sum(differ, 1)
        # 开方得到距离
        distance = np.sqrt(distance)
        # 对距离进行排序，得到的结果为m*1矩阵，内容为当前点到输入数据距离的排名
        sortedIndex = np.argsort(distance, 0)
        # 取距离排序在前k个的所有标签
        for i in range(m):
            # 当取到的点为k时跳出循环
            if i >= self.k:
                break
            index = sortedIndex[i]
            # 获取距离第i个事务的标签
            result.append(self.y[index])
        # 返回结果
        return result
        
    def _judge(self,h):
        '''
        根据knn算出来的k个标签决定该事物应该属于哪个分类
        h 为 k*1向量
        '''
        # 声明一个字典，存放每个类标签的数量
        result_dict = {}
        for i in h:
            result_dict[i] = result_dict.get(i,0) + 1
        # 对字典按值降序排序
        sortedLables = sorted(result_dict.items(), key=lambda result_dict:result_dict[1],reverse=True)
        # 返回排名第一的标签
        return sortedLables[0][0]
    
    def classify(self,inputData):
        '''
        分类方法，inputData为n*1向量
        '''
        # 对事务进行归一化
        normalizedData = self._normalize(inputData)
        # 求出距离最近的k个点
        topK = self._getTopK(normalizedData)
        # 获得k个点中出现次数最多的标签
        result =  self._judge(topK)
        # 返回结果
        return result
        
        
        
        
        
        
