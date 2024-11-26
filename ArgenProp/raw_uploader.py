import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# Prepare the batch_write_item request
def batch_write_items(client, table_name, items):
    request_items = {
        table_name: [
            {"PutRequest": {"Item": item}} for item in items
        ]
    }

    # Call batch_write_item
    response = client.batch_write_item(RequestItems=request_items)
    unprocessed_items = response.get('UnprocessedItems', {})
    
    if unprocessed_items:
        print("Some items were not processed. Retrying...")
        # Retry logic here if needed
        print(unprocessed_items)

    return response

# Leemos el archivo que vamos a querer subir a la tabla
df = pd.read_csv("output/output_argenprop_22112024.csv")
# Hacemos un rename por un error en el scrapper
df.rename(columns={'zonaprop_code': 'argenprop_code'}, inplace=True)

# Dropeamos duplicados del dataframe
print(f"Original data: {df.shape}")
df.drop_duplicates(subset=['argenprop_code'], keep='first', inplace=True)
print(f"Unique data: {df.shape}")

df_data = df.to_dict(orient='records')

load_dotenv()
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_ID'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
)

client = session.client('dynamodb')

processed_df_data = list()
for data in df_data:
    processed_df_data.append({
        'property_url': {'S': str(data['property_url'])},
        'argenprop_code': {'S': str(data['argenprop_code'])},
        'price': {'S': str(data['price'])},
        'expenses': {'S': str(data['expenses'])},
        'address': {'S': str(data['address'])},
        'location': {'S': str(data['location'])},
        'features': {'S': str(data['features'])},
        'description': {'S': str(data['description'])},
        'created_at': {'S': datetime.now().strftime('%Y-%m-%d')},
        'updated_at': {'S': datetime.now().strftime('%Y-%m-%d')},
    })

# Dividimos en chunks de 25 para no superar el límite de DynamoDB
chunks = [processed_df_data[i:i + 25] for i in range(0, len(processed_df_data), 25)]

n=1
for chunk in chunks:
    print(f"Uploading chunk {n} of {len(chunks)}")
    n+=1
    response = batch_write_items(client, 'RAW_ArgenProp', chunk)
    print("Batch write completed. Response:", response)