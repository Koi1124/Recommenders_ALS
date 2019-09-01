# coding=utf-8
"""
@Description: 数据集预处理
先进行ALS处理显式反馈与隐式反馈数据集，得出用户隐因子
再根据用户隐因子进行聚类训练
@Author: Han
@Date: 2019/8/16
"""
import os
from collections import namedtuple

from algorithm.ALSCF import ALSCF
from algorithm.Kmeans import Kmeans
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, ArrayType

BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'sep', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-100k':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path='data/ml-100k/u.data',
            sep='\t',
            reader_params=dict(line_format='userId itemId rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path='data/ml-1m/ratings.dat',
            sep='::',
            reader_params=dict(line_format='userId itemId rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
    'ez_douban':
        BuiltinDataset(
            url='https://pan.baidu.com/s/1DkN1LmdSMzm_jCBKhbPbig',
            path="data/ez_douban/ratings.csv",
            sep=',',
            reader_params=dict(line_format='userId itemId rating timestamp',
                               rating_scale=(1, 5),
                               sep=',')
        )
}


class DataDeal:

    @classmethod
    def load_data(cls, name='ml-100k', spark=0):
        print("正在加载数据集")
        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError("未知数据集：" + name +
                             '\n' + "接受的数据集有：" + ','.join(BUILTIN_DATASETS.keys())
                             )
        if not os.path.isfile(dataset.path):
            raise OSError("数据集：" + name +
                          " 不能在项目data文件夹中找到\n"
                          + "请在" + dataset.url +
                          "下载后放置到data文件下")

        lines = spark.read.text(dataset.path).rdd
        parts = lines.map(lambda row: row.value.split(dataset.sep))

        # itemName = dataset.reader_params.get('line_format').split(' ')[1]

        ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1]),
                                             rating=float(p[2]), timestamp=int(p[3])))
        ratings = spark.createDataFrame(ratingsRDD)
        print(ratings)
        print("数据集加载成功")
        return ratings

    @classmethod
    def offline_data_operate(cls, df_ratings=None, spark=None, isWR=False, trainPercentage=0.8):
        """
        离线操作
        首先对整体数据集进行ALS算法用户特征计算
        根据用户特征进行k-means算法进行聚类
        将得出的k类数据集进行本地的储存
        :param df_ratings: 用户-项目评分的dataframe
        :param spark: 注册的sparkSession
        :param isWR: 是否为隐式反馈
        :param trainPercentage: 训练集百分比
        :return: 返回的为储存在data文件夹中的k类csv
        """
        als = ALSCF(wr=isWR)
        als_model = als.getModel(trainPer=trainPercentage, ratings=df_ratings)
        df_u_p = cls.__feature_transform(model=als_model)
        cls.__total_user_clustering(df_u_p=df_u_p, ratings=df_ratings, spark=spark)

    @staticmethod
    def __feature_transform(model=0):
        print("正在提取用户特征")
        userFactors = model.userFactors.orderBy("id")
        userFactors = userFactors.withColumnRenamed("id", "userId")
        new_schema = ArrayType(FloatType(), containsNull=False)
        udf_foo = udf(lambda x: x, new_schema)
        userFactors = userFactors.withColumn("features", udf_foo("features"))
        kmeans = Kmeans(features=userFactors)
        # 此处的k值由sse算法计算得来
        kmeans.getModel(k=11, seed=4)
        df_transform = kmeans.transform()
        df_transform.write.csv(path="data/user_clustering", mode="overwrite")
        return df_transform

    @staticmethod
    def __total_user_clustering(df_u_p=None, ratings=None, spark=None):
        print('正在进行离线用户聚类')
        ratings.createOrReplaceTempView("ratings")
        for i in range(0, 11):
            list_users = df_u_p.where(df_u_p.prediction == i).collect()
            list_temp = []
            for j in list_users:
                list_temp.append(str(j.userId))
            part = ','.join(list_temp)
            df_ratings_part = spark.sql(
                "SELECT userId, itemId, rating, timestamp FROM ratings WHERE userId in (" + part + ")")
            df_ratings_part.write.csv(path="data/cluster" + str(i), mode="overwrite")
        print("成功完成离线用户聚类")
        # for i in list_users:
        #     df_als_temp = ratings.where(ratings.userId == i.userId)
        #     df_als_part.union(df_als_temp)
        # print(df_als_part.show(5))
        # df_als_part.write.csv("/user_clustering/cluster"+str(1)+".csv")

        # for i in range(1, 12):
        #     list_users = df_u_p.where(df_u_p.prediction == i).collect()
        #     df_als_part = None
        #     # for j in list_users:
        #     #     df_als_temp = df_als_model.where(df_als_model.userId == j.userId)
        #     #     df_als_part.union(df_als_temp)

