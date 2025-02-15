
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

#TODO: add take last one .csv codemod   
OHE_FILE = "tmp/ml_ready_ohe_13022025.csv"
FILE = "tmp/ml_ready_13022025.csv"

# Usamos una expresión regular para capturar la secuencia de 8 dígitos antes de ".csv"
match = re.search(r"_(\d{8})\.csv$", OHE_FILE)
if match:
    LATEST = match.group(1)
else:
    print("No se encontró una fecha en el nombre del archivo.")
    

MODEL_FILE_OUT = "models/model_wo_ohe_"+LATEST+".joblib"
MODEL_FILE_OUT_OHE = "models/model_ohe_"+LATEST+".joblib"

df_wo_ohe = pd.read_csv(FILE)
df_ohe = pd.read_csv(OHE_FILE)

df_wo_ohe.drop(columns=["bathrooms", "garages"], inplace=True)
df_ohe.drop(columns=["bathrooms", "garages"], inplace=True)

from sklearn.model_selection import train_test_split
y = df_wo_ohe.price
X = df_wo_ohe.drop(["price"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Con resultados del CVGridSearch grande

GBoost2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,
                                   max_depth=9, max_features='log2',
                                   min_samples_leaf=2, min_samples_split=5,
                                   loss='squared_error', random_state = 5, subsample=0.5)
GBoost_model2 = GBoost2.fit(X_train, y_train)

GBoost_pred2 = GBoost_model2.predict(X_test)
print("RMSE score is: " + str(mean_squared_error(y_test, GBoost_pred2, squared=False)))
print("Model score is: " + str(GBoost_model2.score(X_test, y_test)))


# Vamos a persistir el modelo que mejor nos dió

import pickle
from joblib import dump, load

dump(GBoost_model2, MODEL_FILE_OUT)


from sklearn.model_selection import train_test_split
y = df_ohe.price
X = df_ohe.drop(["price"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Con resultados del CVGridSearch grande

GBoost2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,
                                   max_depth=9, max_features='log2',
                                   min_samples_leaf=2, min_samples_split=5,
                                   loss='squared_error', random_state = 5, subsample=0.5)
GBoost_model2 = GBoost2.fit(X_train, y_train)

GBoost_pred2 = GBoost_model2.predict(X_test)
print("RMSE score is: " + str(mean_squared_error(y_test, GBoost_pred2, squared=False)))
print("Model score is: " + str(GBoost_model2.score(X_test, y_test)))


# Vamos a persistir el modelo que mejor nos dió

import pickle
from joblib import dump, load

dump(GBoost_model2, MODEL_FILE_OUT_OHE)


