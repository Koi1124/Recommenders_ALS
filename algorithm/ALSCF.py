# coding=utf-8
"""
@Description: ALS算法进行显式或隐式反馈计算
@Author: Han
@Date: 2019/8/19
"""
import shutil

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import os


class ALSCF:
    __model_path = "D:/Model/als_model"

    def __init__(self, rank=10, wr=False, maxInter=5):
        print("正在进行ALS算法")
        self.rank = rank
        self.wr = wr
        self.maxInter = maxInter
        self.__predictions = None
        self.__model = None
        if self.wr:
            self.__model_path = "D:/Model/als_wr_model"

    def getModel(self, trainPer=0.8, ratings=None, isOld=True):
        (train, test) = ratings.randomSplit([trainPer, 1 - trainPer], seed=4)
        if os.path.isdir(self.__model_path) and isOld:
            model = ALSModel.load(self.__model_path)
        else:
            als = ALS(maxIter=self.maxInter, regParam=0.01, userCol="userId", itemCol="itemId", ratingCol="rating",
                      coldStartStrategy="drop", rank=self.rank, seed=4, implicitPrefs=self.wr)
            model = als.fit(train)
        self.__predictions = model.transform(test).select("userId", "itemId", "prediction", "rating").orderBy("userId")
        self.__model = model
        print("完成训练模型")
        return model

    def saveModel(self):
        if self.__model:
            if os.path.isdir(self.__model_path):
                shutil.rmtree(self.__model_path)
            self.__model.save(self.__model_path)
            print("成功储存模型")
        else:
            print("不存在此模型或路径不存在")

    def recommend(self, userId=0):
        recommendations = self.__predictions.where(self.__predictions.userId == userId).select("itemId", "prediction")
        recommendations = recommendations.sort(recommendations.prediction.desc())
        return recommendations

    def evaluate(self):
        if not self.__predictions:
            print("不存在此模型，请先获取模型")
            return
        else:
            evaluator = RegressionEvaluator(labelCol="rating")
            rmse = evaluator.evaluate(self.__predictions)
            print("Root-mean-square error = " + str(rmse))




