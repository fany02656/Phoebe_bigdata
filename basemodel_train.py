import sys

# Other packages
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools
import pandas as pd


def main(spark):
    # train_path = 'train_encode.parquet'
    trains1_path = 'train1_encode.parquet'
    val_path = 'val_encode.parquet'
    test_path='test_encode.parquet'

    # Create dataframes

    train_s1 = spark.read.parquet(trains1_path)
    val = spark.read.parquet(val_path)
    # test=spark.read.parquet(test_path)


    print('load data success')

    # Variables for hyperparameter tuning
    rank=[i for i in range(5, 26, 10)]
    reg_param=[10**i for i in range(-4, 0)]
    # alpha=[i for i in range(10, 31, 10)]
    param_grid = itertools.product(rank, reg_param)

    # rank=[i for i in [5]]
    # reg_param=[10**i for i in [-2]]
    # alpha=[i for i in [5]]
    # best_rmse=7.651545231817701
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")

    # Model training
    # mode_lists=["b1_1"]
    for idx,i in enumerate(param_grid):
        model_s1_name="model_s1_{}".format(idx)
        # print('Start Training for {}'.format(i))
        als = ALS(rank=i[0], regParam=i[1], maxIter=10,
                  userCol="user_i", itemCol="track_i", ratingCol="count",
                  implicitPrefs=True, coldStartStrategy="drop") # The interaction data consists of implicit feedback
        model = als.fit(train_s1)
        prediction = model.transform(val)
        prediction.show(1)
        model.write().overwrite().save(model_s1_name)
        print('successful save  {}'.format(model_s1_name))
        print(f"the model trained with rank {i[0]}, reg_param {i[1]}.")
        # rmse = evaluator.evaluate(prediction)

        # pred_test = model.transform(test)
        # rmse_test = evaluator.evaluate(pred_test)
        # print(f"RMSE of test data is {rmse_test}, and the model trained with rank {i}, reg_param {j}, alpha = {k}.")

        # if rmse < best_rmse:
        #     bestModel = model
        #     best_rmse = rmse
        #     best_rank = i[0]
        #     best_reg = i[1]
        #     best_alpha = i[2]
        #     bestModel.write().overwrite().save(model_file)
        #     print(f"best_rmse: {best_rmse}, best_rank: {best_rank}, best_reg: {best_reg}, best_alpha: {best_alpha}")
        # print('Finish Training for {}'.format(i))

     



# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName("phoebe").config("spark.executor.memory", '30g').config("spark.driver.memory", '30g').getOrCreate()
    main(spark)