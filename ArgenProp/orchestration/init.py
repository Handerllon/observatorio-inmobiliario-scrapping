### System Imports
from datetime import datetime
import os

### Custom Imports
from utils import log, generate_raw_file, generate_stg_file
from scrapping_utils import start_scrapping


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
start_scrapping(OUT_DIR + "/" + STOCK_FILE, PAGES_SCRAPPED)
generate_raw_file(OUT_DIR + "/" + STOCK_FILE, OUT_DIR + "/" + RAW_FILE)
generate_stg_file(OUT_DIR + "/" + RAW_FILE, OUT_DIR + "/" + STG_FILE)


