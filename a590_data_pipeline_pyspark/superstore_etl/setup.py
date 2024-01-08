import os

PWD = os.environ['PGPASS'] # Kami sangat menyarankan Anda untuk menyimpan informasi penting seperti password kedalam environment variabel.
# PWD = "DATABASE_PASSWORD" # Kami mengasumsikan Anda memiliki password yang sama untuk scr database dan target database
USER_ID = "postgres"
SERVER = "db.oqucojswwhlmdptxhcaa.supabase.co"
DB = "postgres"
DRIVER = "org.postgresql.Driver"

SERVER_TARGET = "db.rstxdaajwmqpconknzkv.supabase.co"

# source connection
src_url = f"jdbc:postgresql://{SERVER}:5432/{DB}?user={USER_ID}&password={PWD}"

# target connection
target_url = f"jdbc:postgresql://{SERVER_TARGET}:5432/{DB}?user={USER_ID }&password={PWD}"