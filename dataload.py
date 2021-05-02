from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer

def sub_sample(data_w, data_i, frac=0.01):
    data_w_sub=data_w.sample(withReplacement=False, fraction=frac)
    data_sub=data_i.union(data_w_sub)
    print('Successfully loading {} data'.format(frac))
    data_sub.show(1)
    print(data_sub.count())
    return data_sub



def main(spark):

# Load all the Data
#     train = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
#     train.createOrReplaceTempView('train')
#     print('Successfully load train data')
#     train.show(1)
#     val = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
#     val.createOrReplaceTempView('val')
#     val.show(1)
#     print('Successfully load val data')
#     test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_test.parquet')
#     test.createOrReplaceTempView('test')
#     test.show(1)
#     print('Successfully load test data')
#
# # count three df
#     print('number of data in train: {}, number of data in val: {}, number of \
#     data in test: {}'.format(train.count(), val.count(), test.count()))
#
# #  add another columns user_i, item_i and transform them into int
#
#     users = (train.select('user_id').union(val.select('user_id'))
#             .union(test.select('user_id'))).distinct()
#     tracks = (train.select('track_id').union(val.select('track_id'))
#              .union(test.select('track_id'))).distinct()
#
#     user_encoding = StringIndexer(inputCol="user_id", outputCol="user_i")
#     tran_user = user_encoding.fit(users)
#     track_encoding = StringIndexer(inputCol="track_id", outputCol="track_i")
#     tran_track = track_encoding.fit(tracks)
#
#     ## transform train
#     train = tran_user.transform(train)
#     train = tran_track.transform(train)
#
#     ## transform val
#     val = tran_user.transform(val)
#     val = tran_track.transform(val)
#
#     ## transform test
#     test = tran_user.transform(test)
#     test = tran_track.transform(test)
#
#     train_encode = train.withColumn("user_i", train["user_i"].cast(IntegerType())) \
#         .withColumn("track_i", train["track_i"].cast(IntegerType()))
#     val_encode = val.withColumn("user_i", val["user_i"].cast(IntegerType())) \
#         .withColumn("track_i", val["track_i"].cast(IntegerType()))
#     test_encode = test.withColumn("user_i", test["user_i"].cast(IntegerType())) \
#         .withColumn("track_i", test["track_i"].cast(IntegerType()))
#
#     print('show train_encode')
#     train_encode.show(1)
#     print('show val_encode')
#     val_encode.show(1)
#     print('show test_encode')
#     test_encode.show(1)
#     print('Successfully convert str to int')
#     print(train_encode.dtypes)
#
# # save dataset
#     train_encode.write.parquet('train_encode.parquet')
#     val_encode.write.parquet('val_encode.parquet')
#     test_encode.write.parquet('test_encode.parquet')
#     print('Successfully convert dataset and save')

    
# subsample

    train_path = 'train_encode.parquet'
    val_path = 'val_encode.parquet'
    test_path='test_encode.parquet'
    train_encode = spark.read.parquet(train_path)
    val_encode = spark.read.parquet(val_path)
    test_encode=spark.read.parquet(test_path)
    print('load success')

    t_v_user = val_encode.withColumnRenamed('user_i', 'user_i_').select('user_i_').distinct()
    train_i = train_encode.join(t_v_user, train_encode.user_i == t_v_user.user_i_, how='inner').select('user_i', 'count', 'track_i')
    train_w = train_encode.join(t_v_user, train_encode.user_i == t_v_user.user_i_, how='left_anti').select('user_i', 'count','track_i')

    # subsample 1%
    train1_encode=sub_sample(train_w, train_i, frac=0.01)
    # subsample 5%
    train5_encode=sub_sample(train_w, train_i, frac=0.05)
    # subsample 25%
    train25_encode=sub_sample(train_w, train_i, frac=0.25)

    train1_encode.write.parquet('train1_encode.parquet')
    train5_encode.write.parquet('train5_encode.parquet')
    train25_encode.write.parquet('train25_encode.parquet')
    print('Successfully save subsample')

    # # frac 1%
    # train_w_1 = train_w.sample(withReplacement=False, fraction=0.01)
    # train_1 = train_i.union(train_w_1)
    # print('Successfully loaded 1% data')
    # train_1.show(1)
    # print(train_1.count())

    # #frac 5%
    # train_w_5=train_w.sample(withReplacement=False, fraction=0.05)
    # train_5 = train_i.union(train_w_5)
    # print('Successfully loaded 5% data')
    # train_5.show(1)
    # print(train_5.count())
    #
    #
    # #frac 25%
    # train_w_25=train_w.sample(withReplacement=False, fraction=0.25)
    # train_25 = train_i.union(train_w_25)
    # print('Successfully loaded 25% data')
    # train_25.show(1)
    # print(train_25.count())


if __name__ == "__main__":
    # Create the spark session object
    # Create the spark session object
    spark = SparkSession.builder.appName("phoebe").config("spark.executor.memory", '30g').config("spark.driver.memory", '30g').getOrCreate()
    main(spark)