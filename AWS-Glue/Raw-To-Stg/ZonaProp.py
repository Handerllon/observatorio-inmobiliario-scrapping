from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
import sys

# Inicializar contexto de Glue
args = getResolvedOptions(sys.argv, ['SOURCE_DATABASE', 'SOURCE_TABLE','S3_OUTPUT_PATH'])
def log(level, message):
    levels = ["INFO", "WARNING", "ERROR"]
    if level not in levels:
        raise ValueError(f"Invalid log level: {level}. Use one of {levels}.")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Parámetros
VALOR_DOLAR = -1
NOW = datetime.now().strftime('%Y-%m-%d')

SOURCE_DATABASE = args['SOURCE_DATABASE']
SOURCE_TABLE = args['SOURCE_TABLE']
S3_OUTPUT_PATH = args['S3_OUTPUT_PATH']

print(f"From db.table: {SOURCE_DATABASE}.{SOURCE_TABLE}")
import requests

try:
    res = requests.get("https://dolarapi.com/v1/dolares/blue")
    VALOR_DOLAR = res.json()["compra"]
    log("INFO", f"Dollar value: {VALOR_DOLAR}")
except Exception as e:
    log("ERROR", f"Error in fetching the dollar value: {e}")
    raise RuntimeError(f"Failed to fetch dollar value: {e}")  # ❗ si falla, esto aborta el job

# 1. Leer desde Data Catalog
df = glueContext.create_dynamic_frame.from_catalog(
    database=SOURCE_DATABASE,
    table_name=SOURCE_TABLE
).toDF()
log("INFO", f"Generating RAW file from db:{SOURCE_DATABASE}.{SOURCE_TABLE}")
log("INFO", f"Original data: {df.count()} rows x {len(df.columns)} columns")
df = df.dropDuplicates(["zonaprop_code"])
log("INFO", f"Unique data: {df.count()} rows x {len(df.columns)} columns")
# Save as Parquet
# 8. Escribir en formato Parquet
#processing_date_str = datetime.now().strftime('%Y-%m-%d')
output_path = f"{S3_OUTPUT_PATH}/raw/current_dedup"
df.write.mode("overwrite").parquet(output_path)
log("INFO", f"Saved cleaned data to {output_path} as parquet")
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType
from datetime import datetime

# 1. Separar 'location' en 'neighborhood' y 'location' (sobrescribiendo)
split_loc = F.split(F.col("location"), ",")
df = df.withColumn("neighborhood", F.trim(split_loc.getItem(0)))
df = df.withColumn("location", F.trim(split_loc.getItem(1)))

# 2. Extraer número limpio en 'expenses'
df = df.withColumn("expenses", F.regexp_extract("expenses", r"(\d{1,3}(?:\.\d{3})*)", 1))

# 3. Separar 'price' en 'price_currency' y 'price'
split_price = F.split(F.col("price"), " ")
df = df.withColumn("price_currency", F.trim(split_price.getItem(0)))
df = df.withColumn("price", F.trim(split_price.getItem(1)))

# 4. Filtrar filas donde price == "precio"
df = df.filter(F.trim(F.col("price")) != "precio")

# 5. Si 'price_currency' == 'USD', convertir a pesos
df = df.withColumn(
    "price",
    F.when(
        F.col("price_currency") == "USD",
        F.regexp_replace("price", "\.", "").cast(FloatType()) * VALOR_DOLAR
    ).otherwise(F.col("price").cast(FloatType()))
)

# 6. Eliminar 'price_currency'
df = df.drop("price_currency")

# 7. Extraer campos de 'features'
df = df.withColumn("total_area",   F.regexp_extract("features", r"(\d+)\s?m²", 1).cast(IntegerType()))
df = df.withColumn("rooms",        F.regexp_extract("features", r"(\d+)\s?amb\.?", 1).cast(IntegerType()))
df = df.withColumn("bedrooms",     F.regexp_extract("features", r"(\d+)\s?dorm\.?", 1).cast(IntegerType()))
df = df.withColumn("bathrooms",    F.regexp_extract("features", r"(\d+)\s?bañ(?:os|o)", 1).cast(IntegerType()))
df = df.withColumn("garages",      F.regexp_extract("features", r"(\d+)\s?coch\.?", 1).cast(IntegerType()))

# 8. Eliminar columna 'features'
df = df.drop("features")

# 9. Agregar columna 'antiquity' vacía (NaN)
df = df.withColumn("antiquity", F.lit(None).cast(IntegerType()))

# 10. Timestamp
df = df.withColumn("created_at", F.lit(NOW)) \
       .withColumn("updated_at", F.lit(NOW))

# 8. Escribir en formato Parquet
output_path = f"{S3_OUTPUT_PATH}/stg/current"
log("INFO", f"Escribiendo archivo a {output_path} as parquet")

# archivo del procesamiento diario
df.write.mode("overwrite").parquet(output_path) 
# archivo general, mantiene histórico

general_output_path = f"{S3_OUTPUT_PATH}/stg/all"

# Guardar como parquet particionado por fecha
df.write \
    .mode("append") \
    .format("parquet") \
    .partitionBy("created_at") \
    .save(general_output_path)

log("INFO", f"Saved general data to {general_output_path} as parquet")

# Commit del Job
job.commit()