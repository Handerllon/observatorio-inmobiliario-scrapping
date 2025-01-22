from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np

ZONAPROP_FILE = "../ArgenProp/orchestration/output/STG_ArgenProp_18012025.csv"
ARGENPROP_FILE = "../ZonaProp/orchestration/output/STG_ZonaProp_18012025.csv"

FULL_FILE = "tmp/full_stg_extract_{}.csv".format(str(datetime.now().strftime("%Y-%m-%d")))

df_zonaprop = pd.read_csv(ZONAPROP_FILE)
df_argenprop = pd.read_csv(ARGENPROP_FILE)

df_full = pd.concat([df_zonaprop, df_argenprop], ignore_index=True)
df_full.to_csv(FULL_FILE, index=False)

