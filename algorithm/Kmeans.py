# coding=utf-8
"""
@Description:
@Author: Han
@Date: 2019/8/19
"""
import os
import shutil

from pyspark.ml.clustering import KMeans, KMeansModel


class Kmeans:
    cost = list(range(2, 21))
    __model = None
    __model_path = "D:/Model/kmeans_model"

    def __init__(self, features=None):
        self.features = features

    def check_k(self):
        for k in range(2, 21):
            kmeans = KMeans(k=k, seed=4)
            k_model = kmeans.fit(self.features)
            self.cost[k-2] = k_model.computeCost(self.features)
        print(self.cost)
        for i in self.cost:
            print(i)

    def getModel(self, k=2, seed=4):
        print("正在进行k-聚类算法")
        if os.path.isdir(self.__model_path):
            model = KMeansModel.load(self.__model_path)
        else:
            kmeans = KMeans(k=k, seed=seed)
            model = kmeans.fit(self.features)
        print("完成模型训练")
        self.__model = model
        return model

    def saveModel(self):
        if self.__model:
            if os.path.isdir(self.__model_path):
                shutil.rmtree(self.__model_path)
            self.__model.save(self.__model_path)
            print("成功储存模型")
        else:
            print("不存在此模型或路径不存在")
            return

    def transform(self):
        if self.__model:
            transform = self.__model.transform(self.features).select("userId", "prediction").orderBy("userId")
            return transform
        else:
            print("不存在模型或特征向量")
            return