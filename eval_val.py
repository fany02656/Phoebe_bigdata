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
    val_path="val_encode.parquet"
    val = spark.read.parquet(val_path).select("user_i", "track_i", "count")

    models = ["model_s1_{}".format(idx) for idx in range(12)]
    # results = {"model_s1_{}".format(idx): "Results_ALS_val_{}".format(idx) for idx in range(0)}



    # Evaluation
    for model_file in models:
        model = ALSModel.load(model_file)
        test_users = val.select("user_i").distinct()
        prediction = model.recommendForUserSubset(test_users, 500)
        order_track = Window.partitionBy('user_i').orderBy(col('count').desc())

        ## create true top 500 tracks for each users
        true_top_track = val.select("user_i", "track_i", "count",F.rank().over(order_track).alias('rank'))\
            .where('rank <= 500').groupBy('user_i').agg(expr('collect_list(track_i) as track_true'))
        # true_top_track.show(5)
        # pre_true_join = true_top_track.join(prediction, 'user_i','inner')
        pre_true_join = true_top_track.join(prediction, 'user_i')\
            .rdd.map(lambda row: ([cobine.track_i for cobine in row["recommendations"]], row["track_true"]))
        # print('finish join prediction and true')

        # Ranking metrics
        rankingMetrics = RankingMetrics(pre_true_join)
        precision_at = rankingMetrics.precisionAt(500)
        map = rankingMetrics.meanAveragePrecision
        ndcg = rankingMetrics.ndcgAt(500)
        print("{}:[precision_at:{}, map:{}, ndcg:{}]".format(model_file,precision_at,map,ndcg))

        # val_res = spark.createDataFrame([(precision_at, map, ndcg)], ["Precision_At", "mean_Av_Prec",  "NDCG_At"])
        # val_res.show()
        # val_res.write.mode("overwrite").parquet(results[model_file])


# Only enter this block if we're in main
if __name__ == "__main__":
    spark = SparkSession.builder.appName("phoebe").config("spark.executor.memory", '30g').config("spark.driver.memory", '30g') .getOrCreate()
    main(spark)