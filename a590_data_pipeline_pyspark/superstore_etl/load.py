from setup import *

def load(df, tabel_name):
    """Load transformed data into target database

    Args:
        df (pyspark.sql.dataframe.DataFrame): Transformed data
        tabel_name (str): Tabel name
    """
    try:
        rows_imported = 0
        print(f'importing {df.count()} rows ... for table {tabel_name}')
        df.write.mode("overwrite") \
            .format("jdbc") \
            .option("url", target_url) \
            .option("user", USER_ID) \
            .option("password", PWD) \
            .option("driver", DRIVER) \
            .option("dbtable", tabel_name) \
            .save()
        print("Data imported successful")
        rows_imported += df.count()
        
    except Exception as e:
        print("Data load error: " + str(e))