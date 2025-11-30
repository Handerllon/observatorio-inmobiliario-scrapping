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

PAGES_SCRAPPED = 500

OUT_DIR = "output"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    log("INFO", "Output directory created successfully!")
else:
    log("INFO", "Output directory already exists!")

# Inicializamos el proceso de scraping
#start_scrapping("data/zonaprop/raw/" + RAW_FILE, PAGES_SCRAPPED, s3_client, BUCKET_NAME)
#generate_raw_file("scrapping/ZonaProp/STOCK/" + STOCK_FILE, "scrapping/ZonaProp/RAW/" + RAW_FILE, s3_client, BUCKET_NAME)
# generate_stg_file("scrapping/ZonaProp/RAW/" + RAW_FILE, "scrapping/ZonaProp/STG/" + STG_FILE, s3_client, BUCKET_NAME)

from utils import generate_local_stg_file
# local process
RAW_FILE = "" # local process

import glob
# Local file pattern matching
matching_files = glob.glob("RAW_ZonaProp_*.csv")
if len(matching_files) > 1:
    raise ValueError(f"Error: Multiple files found matching pattern 'RAW_ZonaProp_*.csv': {matching_files}")
elif len(matching_files) == 1:
    RAW_FILE = matching_files[0]
    print(f"Using local file: {RAW_FILE}")
else:
    RAW_FILE = None
    print("No local file matching 'RAW_ZonaProp_*.csv' found, will use S3")

print("Processing file: {}".format(RAW_FILE))

print("WARNING: Tener en cuenta el valor dolar de la fecha del archivo RAW: {}".format(RAW_FILE))
generate_local_stg_file(RAW_FILE)


