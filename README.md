# observatorio-inmobiliario-scrapping

## Instalación

Se requiere Python3 para ejecutar la mayoría de los scripts presentes
Generar entorno virtual
`python3 -m venv env`

Instalar librerías
`python3 -m pip install -r requirements.txt`

## Notas Importantes
Para poder interactuar con el entorno AWS se requiere generar un archivo `.env` y depositar allí las Keys de AWS. Se puede utilizar de referencia el archivo `.env_example`.

Una vez depositadas las keys, se puede ejecutar el archivo `aws_init.py` para generar automáticamente las tablas necesarias para depositar la información.

Para la porción de scrapping es necesario tener instalado `chromium`.

En varios de los archivos encontrarán constantes al principio de los scripts que los ayudarán a configurar algunas de las porciones de ejecución de los mismos.

## Scrappers
Dentro de las carpetas ArgenProp y ZonaProp van a encontrar otra carpeta llamada scrapping. Allí podrán encontrar los scripts para ejecutar la extracción de información para ambos portales inmobiliarios
 
## ETLs
Dentro de las carpetas ArgenProp y ZonaProp van a encontrar una carpeta llamada etls. Aquí van a ver varios archivos que cumplen distintos pasos en el preparado de los datos.
- file_to_raw.py -> Realiza una extracción de los archivos generados por la porción de scrapping y los deposita en la tabla DynamoDB "RAW" de cada portal. Por ejemplo, "RAW_ZonaProp"
- raw_to_stg.py -> Realiza una extracción de, por ejemplo, "RAW_ZonaProp", procesa la información para hacerla mas "linda" (Normaliza precios, realiza calculos de superficie, llena información de baños, cuartos, cocheras, etc) y la deposita en la respectiva tabla "STG". Por ejemplo "STG_ZonaProp"

## Machine Learning
Encontrarán la carpeta Machine Learning. Aquí podrán encontrar lo siguiente:
- stg_extractor.py -> Realiza una extracción de toda la información procesada (Tablas "STG_ZonaProp" y "STG_ArgenProp") y las deposita en un archivo local
- data_clean.ipynb -> Realiza una limpieza final de la información (Remueve nulos, normaliza antiguedades, llena información, etc) y genera un archivo limpio listo para ser utilizado para la porción de Machine Learning.
- data_analysis.ipynb -> A partir de la información limpia, se realizan análisis adicionales de distribuciones, desviaciones, etc. con el objetivo de normalizar la información lo más posible previo a la generación de los modelos.
- model.ipynb -> Archivo donde se generan los modelos. Se eligió aquí ir por dos caminos, uno con la información sin especificación de barrio y otro con OneHotEncoding aplicado al barrio. Los modelos finales pueden encontrarse en la carpeta Models