#!/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014年9月9日

@author: MaWei
'''
import math
import numpy


class KMeans():
    
    def __init__(self,k,sse):
        self.k = k
        self.sse = sse
    
    def _randCenter(self, arr):
        '''
        随机在arr中选取k个点作为初始中心
        arr为m*n维数组，其中m为训练数据集条数，n为训练数据的feature数
        '''
        # 获取数组的副本
        tempArr = numpy.copy(arr)
        # 计算结果的维度为k*n
        dimension = self.k, numpy.size(tempArr, 1)
        # 构建函数的返回值，为k*n数组
        resultArr = numpy.zeros(dimension)
        # 随机打乱数据
        numpy.random.shuffle(tempArr)
        # 选择前k个数据作为初始中心
        for i in range(self.k):
            resultArr[i] = tempArr[i]
        # 返回结果
        return resultArr
        
    
    def __getDistance(self, v1, v2):
        '''计算两个向量的距离'''
        # 计算两个向量的差
        temp = v1 - v2
        # 计算所有feature的平方和后开方
        return math.sqrt(numpy.sum(temp * temp))
    
    def __findClosestCentroid(self, vector, centroid):
        '''
        计算vector与centroid中各向量的距离
        返回最短距离的centroid index
        '''
        # 初始化距离,用于保存点到中心的最小距离
        minDistance = numpy.inf
        # 初始化结果,用户保存最小距离的index
        index = -1
        # 逐一计算点到每个中心的距离,保留最小值及其index
        for i in range(self.k):
            # 计算距离
            distance = self.__getDistance(vector, centroid[i])
            # 如果该距离小于目前的最小距离
            if distance < minDistance:
                # 更新距离
                minDistance = distance
                # 更新结果
                index = i
        # 返回结果
        return index
            
    def _getNewCentroidIndex(self, raw_data, centroid):
        '''
        将原始数据分配到新的聚类中心中去
        返回的结果为m*1数组,每一行记录保存对应的该行事务被分配到的聚类结果
        '''
        # 获取事务总数
        m = raw_data.shape[0]
        # 计算结果的维度为m*1
        dimension = m, 1
        # 构建函数的返回值，为k*n数组
        result = numpy.zeros(dimension)
        # 对每一个事务进行计算,为其分配聚类
        for i in range(m):
            result[i] = self.__findClosestCentroid(raw_data[i], centroid)
        # 返回结果
        return result
    
    def _getNewCentroid(self, raw_data, index, centroid):
        '''
        根据原始数据及其聚类结果,计算新的聚类中心
        '''
        # 获取事务总数
        m = numpy.size(raw_data, 0)
        # 构造结果
        result = numpy.copy(centroid)
        # 对每一个中心进行计算
        for i in range(self.k):
            # 声明一个计数器,用于对该聚类中心下的原始数据进行计数,以计算均值
            count = 0
            # 计算分配到该中心下所有点的均值
            for j in range(m):
                # 如果原始数据属于该聚类中心
                if index[j] == i:
                    # 则计算所有feature的和
                    result[i] = result[i] + raw_data[j]
                    # 相应的计数器+1
                    count += 1
            # 某一个聚类中心下的所有点计算完毕,开始计算均值
            print ('count is ',count)
            if count != 0:
                result[i] /= count
        # 返回新的聚类中心
        return result
    
    def _isConverge(self, previous, current):
        '''
        判断聚类结果是否收敛
        计算两次聚类结果之间的距离
        如果小于SSE则认为收敛
        '''
        # 声明一个变量,用于存储聚类中心之间的距离之和
        distance = 0
        # 计算两次聚类结果相应聚类中心的距离
        for i in range(self.k):
            distance += self.__getDistance(current[i] , previous[i])
        # 计算距离的均值
        distance /= self.k
        # 当距离均值小于SSE时,则认为收敛
        if distance < self.sse:
            return True
        # 否则需要再次进行计算
        else:
            return False
            
        
    def calculateCost(self, raw_data, index, centroid):
        '''
        计算k-means的最终cost
        即每个点到其聚类中心距离之和
        '''
        # 声明distance用于存放最后的距离
        distance = 0
        # 获取事务总数
        m = raw_data.shape[0]
        for i in range(m):
            # 获取事务属性
            data = raw_data[i]
            # 获取事务被分配到的类索引
            dataClass = int(index[i])
            # 获取事务被分配到的聚类中心
            cent = centroid[dataClass]
            # 计算该事务到其聚类中心的距离并累加
            distance += self.__getDistance(data, cent)
        # 返回距离
        return distance
    
    def clusting(self, raw_data):
        # 随机选择K个点作为初始中心
        priviousCentroid = self._randCenter(raw_data)
        # 用于标识聚类是否收敛
        coverge = False
        # 记录迭代次数
        count = 1
        while not coverge:
            index = self._getNewCentroidIndex(raw_data, priviousCentroid)
            centroid = self._getNewCentroid(raw_data, index, priviousCentroid)
            coverge = self._isConverge(priviousCentroid, centroid)
            priviousCentroid = centroid
            print (centroid)
            print ('第' , count , '次迭代')
            count += 1
        return index, centroid
    
    
if __name__ == '__main__':
    
    
    
    
    
#     # 读取原始数据
#     raw = loadDataSet(self.input_file)
#     raw_data = numpy.array(raw)
#     
#     cost = numpy.Inf
#     
#     for i in range(10):
#         currentIndex, currentCentroid = clusting(raw_data)
#         currentCost = calculateCost(raw_data, currentIndex, currentCentroid)
#         if currentCost < cost :
#             cost = currentCost
#             index = currentIndex
#             centroid = currentCentroid
#         print ('Current cost is :' + str(currentCost))
#         print ('The best cost is :' + str(cost))
#     plotResult(raw_data, index)

    pass
