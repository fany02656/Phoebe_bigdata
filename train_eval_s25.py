import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F



def main(spark):
    trains25_path = 'train25_encode.parquet'
    val_path = 'val_encode.parquet'
    test_path='test_encode.parquet'

### Create dataframes

    train_s25 = spark.read.parquet(trains25_path)
    val = spark.read.parquet(val_path).select("user_i", "track_i", "count")
    test_users = val.select("user_i").distinct()
    order_track = Window.partitionBy('user_i').orderBy(col('count').desc())
    # create true top 500 tracks for each users
    true_top_track = val.select("user_i", "track_i", "count",F.rank().over(order_track).alias('rank'))\
            .where('rank <= 500').groupBy('user_i').agg(expr('collect_list(track_i) as track_true'))


### Variables for hyperparameter tuning
    rank=[i for i in range(5, 26, 10)]
    reg_param=[10**i for i in range(-4, 0)]
    param_grid = itertools.product(rank, reg_param)

### train and evaluate models
    for idx,i in enumerate(param_grid):
        model_name = "model_s25_{}".format(idx)
        ## train the model
        als = ALS(rank=i[0], regParam=i[1], maxIter=10,
                  userCol="user_i", itemCol="track_i", ratingCol="count",
                  implicitPrefs=True, coldStartStrategy="drop") # The interaction data consists of implicit feedback
        model = als.fit(train_s25)
        print(f"the {model_name} trained with rank {i[0]}, reg_param {i[1]}.")
        ## evaluate the model
        # predict top 500
        prediction = model.recommendForUserSubset(test_users, 500)
        # join prediction and true for each user
        pre_true_join = true_top_track.join(prediction, 'user_i') \
            .rdd.map(lambda row: ([cobine.track_i for cobine in row["recommendations"]], row["track_true"]))
        ## Ranking metrics
        rankingMetrics = RankingMetrics(pre_true_join)
        precision_at = rankingMetrics.precisionAt(500)
        map = rankingMetrics.meanAveragePrecision
        ndcg = rankingMetrics.ndcgAt(500)
        print("{}:[precision_at:{}, map:{}, ndcg:{}]".format(model_name, precision_at, map, ndcg))

    

if __name__ == "__main__":
    spark = SparkSession.builder.appName("phoebe").config("spark.executor.memory", '30g').config("spark.driver.memory", '30g') .getOrCreate()
    main(spark)