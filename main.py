# coding=utf-8
"""
@Description: 进行测试
@Author: Han
@Date: 2019/8/14
"""

from algorithm.ALSCF import ALSCF
from algorithm.Kmeans import Kmeans
from calculate.recommend import Recommend
from pyspark.sql import SparkSession
from dataset.data_predeal import DataDeal
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.recommendation import ALS
# from pyspark.sql import Row

if __name__ == '__main__':

    spark = SparkSession.builder.appName("ALS_TEST").getOrCreate()

    # dataset_name = 'ml-100k'
    #
    # ratings = DataDeal.load_data(dataset_name, spark)
    #
    # als = ALSCF(wr=False)
    # model = als.getModel(trainPer=0.7, ratings=ratings)
    # DataDeal.total_user_clustering(df_u_p=df_u_p, ratings=ratings, spark=spark)

    recommenders = Recommend(spark=spark, isWR=False)
    # recommenders.evaluate(userId=5)
    # recommenders.recommend_item(userId=5, count=5)
    recommenders.recommend_item_streaming(userId=5)

    # dataset_name = 'ml-100k'
    # ratings = DataDeal.load_data(dataset_name, spark)
    # als = ALSCF(wr=False)
    # als.getModel(ratings=ratings)
    # als.evaluate()


    # recommenders.temp(userId=2)
    # recommenders.data_stream_deal(userId=2)

    # prediction = model.transform(test)
    #
    # evaluator = RegressionEvaluator(metricName="rmse", predictionCol="prediction", labelCol="rating")
    #
    # rmse = evaluator.evaluate(prediction)
    #
    # print("Root-mean-square error = " + str(rmse))
    #
    # userRecs = model.recommendForAllUsers(10)
    #
    # movieRecs = model.recommendForAllItems(10)
    #
    # userRecs.show()
    #
    # movieRecs.show()

