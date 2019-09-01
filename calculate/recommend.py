# coding=utf-8
"""
@Description:
@Author: Han
@Date: 2019/8/20
"""
from algorithm.ALSCF import ALSCF
from pyspark.sql.types import FloatType, StructType, StructField, IntegerType
from pyspark.streaming import StreamingContext


class Recommend:
    __spark = None
    __u_type = 0
    __user_id = 0

    def __init__(self, spark=None, isWR=False):
        self.__spark = spark
        self.__isWR = isWR
        self.__als = ALSCF(wr=isWR)

    def __get_transformed(self):
        schema = StructType([
            StructField("userId", IntegerType()),
            StructField("prediction", IntegerType())
        ])
        df_transformed = self.__spark.read.csv(path="data/user_clustering", schema=schema)
        return df_transformed

    def __user_clustering(self, userId=0):
        if (not self.__u_type) or (userId != self.__user_id):
            self.__user_id = userId
            df_u_p = self.__get_transformed()
            print("正在为用户进行聚类")
            list_temp = df_u_p.where(df_u_p.userId == userId).collect()
            u_type = list_temp[0].prediction
            print(u_type)
            self.__u_type = u_type
        schema = StructType([
            StructField("userId", IntegerType()),
            StructField("itemId", IntegerType()),
            StructField("rating", FloatType()),
            StructField("timestamp", IntegerType())
        ])
        df_dataset = self.__spark.read.csv(path="data/cluster" + str(self.__u_type), schema=schema)
        return df_dataset

    def recommend_item(self, userId=0, count=0):
        ratings = self.__user_clustering(userId=userId)
        self.__als.getModel(trainPer=0.8, ratings=ratings, isOld=False)
        recommendations = self.__als.recommend(userId=userId)
        recommendations.show(count)

    def __recommend_item_helper(self, time, rdd):
        ratings = self.__user_clustering(userId=self.__user_id)
        self.__als.getModel(trainPer=0.8, ratings=ratings, isOld=False)
        recommendations = self.__als.recommend(userId=self.__user_id)
        recommendations.show(5)

    def log_deal(self, t, rdd):
        if self.__isWR:
            """
            进行对应的隐式反馈日志分析
            根据相应规定进行加权算分
            """
        else:
            """
            进行对应的显式反馈日志分析
            """

    def recommend_item_streaming(self, userId=0):
        self.__user_id = userId
        sc = self.__spark.sparkContext
        ssc = StreamingContext(sc, 20)
        path = "data/cluster"+str(self.__u_type)
        lines = ssc.textFileStream(path)
        lines.foreachRDD(func=self.log_deal)
        lines.foreachRDD(func=self.__recommend_item_helper)

        ssc.start()
        ssc.awaitTermination()

    def evaluate(self, userId=0):
        ratings = self.__user_clustering(userId=userId)
        self.__als.getModel(trainPer=0.8, ratings=ratings, isOld=False)
        self.__als.evaluate()

