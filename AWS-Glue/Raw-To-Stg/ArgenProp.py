from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from awsglue.job import Job

#sc = SparkContext()
#glueContext = GlueContext(sc)
#spark = glueContext.spark_session
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Parámetros
SOURCE_DATABASE = "zonaprop_db"
SOURCE_TABLE = "raw"
VALOR_DOLAR = 1300
NOW = datetime.now().strftime('%Y-%m-%d')
S3_OUTPUT_PATH = f"s3://../data/zonaprop/stg/"

# 1. Leer desde Data Catalog
df = glueContext.create_dynamic_frame.from_catalog(
    database=SOURCE_DATABASE,
    table_name=SOURCE_TABLE
).toDF()

# 2. Separar 'location' en 'neighborhood' y 'location'
df = df.withColumn("neighborhood", F.split("location", ",").getItem(0)) \
       .withColumn("location", F.lit("Capital Federal"))

# 3. Limpiar 'expenses'
df = df.withColumn("expenses", F.regexp_extract("expenses", r'(\d{1,3}(?:\.\d{3})*)', 1))

# 4. Separar 'price_currency' y 'price'
df = df.withColumn("price_currency", F.split("price", " ").getItem(0)) \
       .withColumn("price", F.split("price", " ").getItem(1))

# 5. Filtrar precios válidos
df = df.filter(df["price"].isNotNull() & (F.trim(df["price"]) != "precio"))

# 6. Convertir a pesos si está en USD
df = df.withColumn("price", F.when(
    F.col("price_currency") == "USD",
    F.col("price").cast("double") * VALOR_DOLAR
).otherwise(F.col("price").cast("double")))

df = df.drop("price_currency")

# 7. Extraer features
df = df.withColumn("total_area", F.regexp_extract("features", r"(\d+)\s?m²", 1).cast(DoubleType())) \
       .withColumn("rooms", F.regexp_extract("features", r"(\d+)\s?amb\.?", 1).cast(DoubleType())) \
       .withColumn("bedrooms", F.regexp_extract("features", r"(\d+)\s?dorm\.?", 1).cast(DoubleType())) \
       .withColumn("bathrooms", F.regexp_extract("features", r"(\d+)\s?bañ(?:os|o)", 1).cast(DoubleType())) \
       .withColumn("garages", F.regexp_extract("features", r"(\d+)\s?coch\.?", 1).cast(DoubleType()))

df = df.drop("features")

# 8. Agregar antigüedad
df = df.withColumn("antiquity", F.lit(None).cast(DoubleType()))

# 9. Timestamp
df = df.withColumn("created_at", F.lit(NOW)) \
       .withColumn("updated_at", F.lit(NOW))

# 10. Escribir en S3 como Parquet (zona STG)
df.write.mode("overwrite").parquet(S3_OUTPUT_PATH)

# Guardar como parquet particionado por fecha
#df.write \
#    .mode("overwrite") \
#    .format("parquet") \
#    .partitionBy("fecha") \
#    .save(S3_OUTPUT_PATH)

# Commit del Job
job.commit()