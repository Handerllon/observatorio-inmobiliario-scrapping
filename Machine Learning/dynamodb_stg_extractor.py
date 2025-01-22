# Script para traer toda la información de las tablas de DynamoDB
# y depositarlas en un archivo
import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np

ZONAPROP_TABLE_NAME = "STG_ZonaProp"
ARGENPROP_TABLE_NAME = "STG_ArgenProp"

TMP_ZONAPROP_FILE = "tmp/zonaprop_stg_extract_{}.csv".format(str(datetime.now().strftime("%Y-%m-%d")))
TMP_ARGENPROP_FILE = "tmp/argenprop_stg_extract_{}.csv".format(str(datetime.now().strftime("%Y-%m-%d")))
FULL_FILE = "tmp/full_stg_extract_{}.csv".format(str(datetime.now().strftime("%Y-%m-%d")))

def get_all_items_from_table(client, table_name):
    """
    Retrieves all items from the specified DynamoDB table.
    """
    items = []
    response = client.scan(TableName=table_name)

    # Collect the first batch of items
    items.extend(response['Items'])

    # Continue fetching items if there are more
    while 'LastEvaluatedKey' in response:
        response = client.scan(
            TableName=table_name,
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        items.extend(response['Items'])
    
    return items

def dynamodb_items_to_dataframe(items):
    """
    Converts a list of DynamoDB items into a Pandas DataFrame.
    """
    # Convert DynamoDB format to Python dictionaries
    data = [{k: list(v.values())[0] for k, v in item.items()} for item in items]
    
    # Create a Pandas DataFrame
    return pd.DataFrame(data)

load_dotenv()
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
)
client = session.client('dynamodb')

print("Fetching data from DynamoDB table {}...".format(ZONAPROP_TABLE_NAME))
items_zonaprop = get_all_items_from_table(client, ZONAPROP_TABLE_NAME)
print("Converting items to DataFrame...")
df_zonaprop = dynamodb_items_to_dataframe(items_zonaprop)
print("Data loaded successfully.")
df_zonaprop.to_csv(TMP_ZONAPROP_FILE, index=False)

print("Fetching data from DynamoDB table {}...".format(ARGENPROP_TABLE_NAME))
items_argenprop = get_all_items_from_table(client, ARGENPROP_TABLE_NAME)
print("Converting items to DataFrame...")
df_argenprop = dynamodb_items_to_dataframe(items_argenprop)
print("Data loaded successfully.")
df_argenprop.to_csv(TMP_ARGENPROP_FILE, index=False)

df_merged = pd.concat([df_argenprop, df_zonaprop], ignore_index=True, sort=False)
df_merged.to_csv(FULL_FILE, index=False)