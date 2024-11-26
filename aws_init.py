import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

def create_table_if_not_exists(client, table_name, partition_key):
    try:
        # Check if the table exists
        response = client.describe_table(TableName=table_name)
        print(f"Table '{table_name}' already exists.")
        return response['Table']
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"Table '{table_name}' not found. Creating it...")
            # Create the table
            table = client.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": partition_key['name'], "KeyType": "HASH"}
                ],
                AttributeDefinitions=[
                    {"AttributeName": partition_key['name'], "AttributeType": partition_key['type']}
                ],
                ProvisionedThroughput={
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5
                }
            )
            print(f"Table '{table_name}' is being created. Please wait until it's active.")
            return table
        else:
            raise

tables_to_create = [
    {"name": "RAW_ZonaProp", "partition_key": {"name": "zonaprop_code", "type": "S"}},
    {"name": "STG_ZonaProp", "partition_key": {"name": "zonaprop_code", "type": "S"}},
    {"name": "RAW_ArgenProp", "partition_key": {"name": "argenprop_code", "type": "S"}},
    {"name": "STG_ArgenProp", "partition_key": {"name": "argenprop_code", "type": "S"}},
]

load_dotenv()
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
)

client = session.client('dynamodb')

# Iterate and create tables if they don't exist
if __name__ == "__main__":
    for table in tables_to_create:
        print(f"Processing table: {table['name']}")
        table_info = create_table_if_not_exists(
            client,
            table_name=table['name'],
            partition_key=table['partition_key']
        )
        print(f"Table info for '{table['name']}':", table_info)
