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
import sys
import pyspark
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
# Other packages
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F

def main(spark):
   model_file="hdfs://horton.hpc.nyu.edu:8020/user/xy2122/val_encode.parquet/b1_0"
   #test_path="hdfs://horton.hpc.nyu.edu:8020/user/xy2122/test_encode.parquet"
   test=spark.read.parquet(f"hdfs://horton.hpc.nyu.edu:8020/user/xy2122/test_encode.parquet")
   test.show(1)
   model = ALSModel.load(model_file)
   print('sss')

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName("phoebe").config("spark.executor.memory", '30g').config("spark.driver.memory", '30g').getOrCreate()
    main(spark)