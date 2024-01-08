from extract import extract
from transform import transform
from load import load
from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .appName('ETLPipeline') \
        .config("spark.jars", "postgresql-42.6.0.jar") \
        .getOrCreate()

sql = "select * from orders"
df = extract(spark=spark, sql=sql)
orders_df, rfm_df = transform(df)
# print(orders_df.show(5))
# print(rfm_df.show(5))

load(df=orders_df, tabel_name="orders")
load(df=rfm_df, tabel_name="rfm")