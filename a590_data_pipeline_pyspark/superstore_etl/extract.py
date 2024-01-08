from setup import *

def extract(spark, sql):
    """Extracting data from database source

    Args:
        spark (pyspark.sql.session.SparkSession): PySpark session
        sql (str): SQL query

    Returns:
        pyspark.sql.dataframe.DataFrame: Extracted data
    """
    try:
        df = spark.read \
            .format("jdbc") \
            .options(driver='org.postgresql.Driver', 
                    user=USER_ID, 
                    password=PWD, 
                    url=src_url, 
                    query=sql) \
            .option("inferSchema", "true") \
            .option("header", "true") \
            .load()
            
        return df
    
    except Exception as e:
        print("Data extract error: " + str(e))

if __name__ == "__main__":
    
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName('ETLPipeline') \
        .config("spark.jars", "postgresql-42.6.0.jar") \
        .getOrCreate()
    sql = "select * from orders"
    df = extract(spark=spark, sql=sql)
    print(df.show(5))