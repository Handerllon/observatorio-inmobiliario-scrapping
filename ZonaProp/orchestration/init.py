### System Imports
from datetime import datetime
import os

### Custom Imports
from utils import log, generate_raw_file, generate_stg_file
from scrapping_utils import start_scrapping
import boto3
from dotenv import load_dotenv

load_dotenv()
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID')
)
s3_client = session.client('s3')

BUCKET_NAME = os.getenv('BUCKET_NAME')

### Script orquestrador para todo el proceso de
### extracción de datos de ZonaProp
date = datetime.now().strftime('%d%m%Y')
STOCK_FILE = f"STOCK_ZonaProp_{date}.csv"
RAW_FILE = f"RAW_ZonaProp_{date}.csv"
STG_FILE = f"STG_ZonaProp_{date}.csv"

PAGES_SCRAPPED = 10

OUT_DIR = "output"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    log("INFO", "Output directory created successfully!")
else:
    log("INFO", "Output directory already exists!")

# Inicializamos el proceso de scraping
start_scrapping("scrapping/ZonaProp/STOCK/" + STOCK_FILE, PAGES_SCRAPPED, s3_client, BUCKET_NAME)
generate_raw_file("scrapping/ZonaProp/STOCK/" + STOCK_FILE, "scrapping/ZonaProp/RAW/" + RAW_FILE, s3_client, BUCKET_NAME)
generate_stg_file("scrapping/ZonaProp/RAW/" + RAW_FILE, "scrapping/ZonaProp/STG/" + STG_FILE, s3_client, BUCKET_NAME)


