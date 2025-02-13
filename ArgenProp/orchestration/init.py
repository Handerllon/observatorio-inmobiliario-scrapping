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
### extracción de datos de ArgenProp
date = datetime.now().strftime('%d%m%Y')
STOCK_FILE = f"STOCK_ArgenProp_{date}.csv"
RAW_FILE = f"RAW_ArgenProp_{date}.csv"
STG_FILE = f"STG_ArgenProp_{date}.csv"

PAGES_SCRAPPED = 500

OUT_DIR = "output"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    log("INFO", "Output directory created successfully!")
else:
    log("INFO", "Output directory already exists!")

# Inicializamos el proceso de scraping
# Inicializamos el proceso de scraping
start_scrapping("scrapping/ArgenProp/STOCK/" + STOCK_FILE, PAGES_SCRAPPED, s3_client, BUCKET_NAME)
generate_raw_file("scrapping/ArgenProp/STOCK/" + STOCK_FILE, "scrapping/ArgenProp/RAW/" + RAW_FILE, s3_client, BUCKET_NAME)
generate_stg_file("scrapping/ArgenProp/RAW/" + RAW_FILE, "scrapping/ArgenProp/STG/" + STG_FILE, s3_client, BUCKET_NAME)


