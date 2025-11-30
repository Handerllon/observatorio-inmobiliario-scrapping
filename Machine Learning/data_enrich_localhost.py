import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import boto3
import os
from io import StringIO
from datetime import datetime

date = datetime.now().strftime('%d%m%Y')
#today

full_fule = "cleaned_zonaprop_argenprop_30112025.csv"
print("Using input file {}".format(full_fule))


# Obtenemos los últimos full de base de información
def extract_date(file):
    date_part = file.split('_')[-1].replace('.csv', '')  # Get "04022025"
    return pd.to_datetime(date_part, format="%d%m%Y")  # Convert to datetime

date = extract_date(full_fule)
FINAL_FILE = f"ml_ready_{date}.csv"
FINAL_FILE_OHE = f"ml_ready_ohe_{date}.csv"
# OUTPUT_FILE = "machine_learning/data/ml_ready/" + FINAL_FILE
# OUTPUT_FILE_OHE = "machine_learning/data/ml_ready_ohe/" + FINAL_FILE_OHE



df = pd.read_csv(full_fule)

# Primero realizamos una limpieza de valores que no tienen sentido o
# no nos servirían para la porción de Machine Learning

# El precio tiene que ser mayor a 0 obligatoriamente
print("Registros antes de borrar los precios menores a 1000: ", len(df))
df = df[df["price"] > 1000]
print("Registros después de borrar los precios menores a 1000: ", len(df))

# El área total debe ser mayor a 0 obligatoriamente
print("Registros antes de borrar las superficies menores a 10: ", len(df))
df = df[df["total_area"] > 10]
print("Registros después de borrar las superficies menores a 10: ", len(df))

# Vemos que el precio tiene una distribución asimétrica a la derecha
# Vamos a ahora quitar los outliers en relacion al precio
print(f"Limpieza precio")

# Calculamos los cuantiles
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Calculamos los límites a partir de los cuantiles
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Linea de borrado mínima': {lower_bound}")
print(f"Linea de borrado máxima': {upper_bound}")

# identificamos los outliers y luego los borramos
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"\nSe detectó la siguiente cantidad de outliers: {outliers.shape[0]}")

# Remove outliers from the DataFrame
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

print("\nNueva cantidad de datos luego de borrado:")
print(df.shape)

# Tenemos un problema similar con la superficie total. Tomamos un approach similar

# Vemos que el precio tiene una distribución asimétrica a la derecha
# Vamos a ahora quitar los outliers en relacion al precio

# Calculamos los cuantiles
print(f"Limpieza superficie total")

Q1 = df['total_area'].quantile(0.25)
Q3 = df['total_area'].quantile(0.75)
IQR = Q3 - Q1

# Calculamos los límites a partir de los cuantiles
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Linea de borrado mínima': {lower_bound}")
print(f"Linea de borrado máxima': {upper_bound}")

# identificamos los outliers y luego los borramos
outliers = df[(df['total_area'] < lower_bound) | (df['total_area'] > upper_bound)]
print(f"\nSe detectó la siguiente cantidad de outliers: {outliers.shape[0]}")

# Remove outliers from the DataFrame
df = df[(df['total_area'] >= lower_bound) & (df['total_area'] <= upper_bound)]

print("\nNueva cantidad de datos luego de borrado:")
print(df.shape)


# Vamos a hacer dos outputs para la porción de Machine Learning

# Output 1: Guardamos el dataset limpio
# borramos la columna de barrio
df_simple = df.drop(columns=['neighborhood'])
# Enviamos info a un archivo csv para trabajar en el siguiente paso
csv_buffer_simple = FINAL_FILE
df_simple.to_csv(csv_buffer_simple, index=False)
print("Cleaned data saved to {}".format(csv_buffer_simple))

# Output 2: Guardamos el dataset limpio con one hot encoding
df_ohe = pd.get_dummies(df, columns=['neighborhood'])
# Enviamos info a un archivo csv para trabajar en el siguiente paso
csv_buffer_ohe = FINAL_FILE_OHE
df_ohe.to_csv(csv_buffer_ohe, index=False)
print("OHE cleaned data saved to {}".format(csv_buffer_ohe))