from pyspark.sql.functions import *
from pyspark.sql.window import Window

def rfm(df):
    """Preparing data for RFM analysis

    Args:
        df (pyspark.sql.dataframe.DataFrame): Extracted data

    Returns:
        pyspark.sql.dataframe.DataFrame: Transformed data for RFM analysis
    """
    
    try:
        rfm_df = df.groupBy("Customer ID").agg(
            max("Order Date").alias("last_order_date"),
            countDistinct("Order ID").alias("frequency"),
            sum("Sales").alias("monetary"),
        )
        rfm_df = rfm_df.withColumn("recency", datediff(current_date(), col("last_order_date")))
        
        recency_window = Window.orderBy(col("recency").desc())
        rfm_df = rfm_df.withColumn("r_score", percent_rank().over(recency_window))

        frequency_window = Window.orderBy(col("frequency").asc())
        rfm_df = rfm_df.withColumn("f_score", percent_rank().over(frequency_window))

        monetary_window = Window.orderBy(col("monetary").asc())
        rfm_df = rfm_df.withColumn("m_score", percent_rank().over(monetary_window))
            
        rfm_df = rfm_df.withColumn("rfm_score", (0.2*col("r_score") + 0.3*col("f_score") + 0.5*col("m_score")))
        rfm_df = rfm_df.sort("rfm_score", ascending=False)
        
        return rfm_df
    
    except Exception as e:
        print("RFM transform error: " + str(e))

def transform(df):
    """Transforming raw data

    Args:
        df (pyspark.sql.dataframe.DataFrame): Extracted data

    Returns:
        pyspark.sql.dataframe.DataFrame: Transformed data
    """
    try:
        new_orders_df = df
        new_orders_df = new_orders_df.withColumn("Row ID", col("Row ID").cast("int"))
        new_orders_df = new_orders_df.withColumn("Postal Code", col("Postal Code").cast("string"))
        new_orders_df = new_orders_df.withColumn("Quantity", col("Postal Code").cast("int"))
        new_orders_df = new_orders_df.withColumn("Order Date", to_date(col("Order Date"), "M/d/yyyy"))
        new_orders_df = new_orders_df.withColumn("Ship Date", to_date(col("Order Date"), "M/d/yyyy"))
        
        rfm_df = rfm(new_orders_df)
        
        return new_orders_df, rfm_df
    except Exception as e:
        print("Data transform error: " + str(e))

