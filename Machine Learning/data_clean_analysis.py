import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_FILE = "tmp/full_stg_extract_2025-01-22cleaned.csv"
OUT_OHE_FILE = "tmp/ml_ready_ohe_2025-01-22.csv"
OUT_FILE = "tmp/ml_ready_2025-01-22.csv"

df = pd.read_csv(INPUT_FILE)

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
df_simple.to_csv(OUT_FILE, index=False)

# Output 2: Guardamos el dataset limpio con one hot encoding
df_ohe = pd.get_dummies(df, columns=['neighborhood'])
# Borramos la columna de barrio
df_ohe.to_csv(OUT_OHE_FILE, index=False)


