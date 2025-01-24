from joblib import load
import sklearn
print(sklearn.__version__)
# scikit-learn==1.5.2
# numpy==2.1.3
# scipy==1.14.1

# Ruta al modelo guardado
model_path = 'models/model_wo_ohe_2024-12-01.joblib' # 6 features
#model_path = 'models/model_wo_ohe_2024-12-04.joblib' # 4 features

# Cargar el modelo desde el archivo
model = load(model_path)

# Supongamos que tienes un vector de entrada para la predicción
input_features = [[26.0,1.0,0.0,1.0,1.0,38.0]]  # Ejemplo de entrada

# Realizar la predicción
prediction = model.predict(input_features)

# Imprimir el resultado
print("Predicción:", prediction)
