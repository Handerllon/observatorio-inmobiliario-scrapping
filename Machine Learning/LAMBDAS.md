# Lambdas de Machine Learning - Guía Completa

Este documento describe todas las funciones Lambda del proyecto, cómo deployarlas y troubleshooting común.

## Tabla de Contenidos
- [Arquitectura General](#arquitectura-general)
- [Lambdas Disponibles](#lambdas-disponibles)
- [Problema: SAM no detecta Docker en macOS](#problema-sam-no-detecta-docker-en-macos)
- [Deploy Manual Workflow](#deploy-manual-workflow)
- [Testing y Verificación](#testing-y-verificación)
- [Troubleshooting](#troubleshooting)

---

## Arquitectura General

Todas las Lambdas usan:
- **Package Type**: Container Image (Docker)
- **Arquitectura**: x86_64 (importante para Mac ARM/M1/M2)
- **Registry**: Amazon ECR
- **IaC**: AWS SAM (Serverless Application Model)

---

## Lambdas Disponibles

### 1. Inference OHE - Python 3.13 (Producción)

**Función**: Inferencia de precios con One-Hot Encoding
**Stack**: `valorar-inference`
**Lambda**: `valorar-inference-ohe-PX2BhKiczrDk`
**Directorio**: `aws-docker-inference-ohe/`
**ECR Repo**: `valorarinferenceb37939bd/ohe3ba8254brepo`
**Image Tag**: `ohe-4a70cb417aa9-python3.13`
**Runtime**: Python 3.13

**Variables de entorno:**
- `BUCKET_NAME`: `observatorio-inmobiliario-v2`

**Configuración:**
- Memory: 256 MB
- Timeout: 60 segundos

---

### 2. Inference OHE - Python 3.12 (Nueva)

**Función**: Inferencia de precios con One-Hot Encoding (versión Python 3.12)
**Stack**: `valorar-inference-py12`
**Lambda**: `valorar-inference-py12-ohe-sUc6pJx4Qirq`
**Directorio**: `aws-docker-inference-ohe-py12/`
**ECR Repo**: `valorar-inference-ohe-py12`
**Image Tag**: `latest`
**Runtime**: Python 3.12

**Variables de entorno:**
- `BUCKET_NAME`: `observatorio-inmobiliario`

**Configuración:**
- Memory: 256 MB
- Timeout: 60 segundos

**Dependencias principales:**
```
scikit-learn==1.7.2
pandas
numpy
boto3==1.28.2
```

---

### 3. Inference OHE By Neighborhood

**Stack**: `valorar-inference-ohe-ByNeighborhood`
**Lambda**: `valorar-inference-ohe-ByNeighborhood-4lTUzUA5OdoE`
**Función**: Inferencia por barrio específico

---

### 4. Training

**Stack**: `valorar-train`
**Lambda**: `valorar-train-train-3WIo3ByVJb4Q`
**Función**: Entrenamiento de modelos ML

---

### 5. Reporting

**Stack**: `valorar-report`
**Lambda**: `valorar-report-report-AYba9EqmJSub`
**Función**: Generación de reportes

---

## Problema: SAM no detecta Docker en macOS

### Síntoma
```bash
sam build
# Error: Building image for ohe requires Docker. is Docker running?
```

### Causa
SAM CLI tiene problemas detectando el socket de Docker en macOS, especialmente en sistemas con Apple Silicon (M1/M2).

### Solución
**No usar `sam build`**. En su lugar, usar build manual con Docker + push a ECR + deploy con SAM.

Cada directorio Lambda tiene un script `deploy-manual.sh` que automatiza este proceso.

---

## Deploy Manual Workflow

### Proceso General

Todos los deploys siguen este patrón:

```bash
# 1. Build de imagen Docker (forzando arquitectura x86_64)
docker build --platform linux/amd64 -t <IMAGE_NAME> .

# 2. Login a ECR
aws ecr get-login-password --region us-east-2 --profile uade-valorar | \
  docker login --username AWS --password-stdin 650532183679.dkr.ecr.us-east-2.amazonaws.com

# 3. Tag de la imagen
docker tag <IMAGE_NAME>:latest <ECR_REPO>:latest

# 4. Push a ECR
docker push <ECR_REPO>:latest

# 5. Deploy con SAM (sin build)
sam deploy --stack-name <STACK_NAME> --region us-east-2 --capabilities CAPABILITY_IAM --resolve-s3
```

### Script Automatizado

Cada directorio tiene `deploy-manual.sh`:

```bash
cd "Machine Learning/aws-docker-inference-ohe-py12"
./deploy-manual.sh
```

### Ejemplo: Deploy de Lambda Python 3.12

```bash
cd "Machine Learning/aws-docker-inference-ohe-py12"

# El script hace todo automáticamente:
# - Build con arquitectura correcta
# - Login a ECR
# - Tag y push
# - Deploy con SAM
./deploy-manual.sh
```

---

## Testing y Verificación

### Listar todas las Lambdas

```bash
export AWS_PROFILE=uade-valorar
aws lambda list-functions --region us-east-2 \
  --query "Functions[?contains(FunctionName, 'valorar')].{Name:FunctionName, Runtime:PackageType, Updated:LastModified}" \
  --output table
```

### Ver detalles de una Lambda específica

```bash
export AWS_PROFILE=uade-valorar
aws lambda get-function \
  --function-name valorar-inference-py12-ohe-sUc6pJx4Qirq \
  --region us-east-2 \
  --query '{Name:Configuration.FunctionName, Memory:Configuration.MemorySize, Timeout:Configuration.Timeout, Image:Code.ImageUri, Env:Configuration.Environment}' \
  --output json
```

### Invocar Lambda desde CLI

```bash
export AWS_PROFILE=uade-valorar
aws lambda invoke \
  --function-name valorar-inference-py12-ohe-sUc6pJx4Qirq \
  --region us-east-2 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"total_area": 50, "rooms": 2, "bedrooms": 1, "antiquity": 10, "neighborhood": "PALERMO"}' \
  /tmp/response.json

cat /tmp/response.json
```

### Ver logs en tiempo real

```bash
export AWS_PROFILE=uade-valorar
aws logs tail /aws/lambda/valorar-inference-py12-ohe-sUc6pJx4Qirq \
  --region us-east-2 \
  --follow
```

### Ver logs recientes (últimos 5 minutos)

```bash
export AWS_PROFILE=uade-valorar
aws logs tail /aws/lambda/valorar-inference-py12-ohe-sUc6pJx4Qirq \
  --region us-east-2 \
  --since 5m
```

---

## Troubleshooting

### Error: "list index out of range"

**Causa**: No hay modelos en S3 que cumplan los criterios de búsqueda.

**Solución**:
1. Verificar que existan archivos `.joblib` en el bucket
2. Revisar que estén en la carpeta correcta (`models/`)
3. Verificar el naming pattern esperado

```bash
export AWS_PROFILE=uade-valorar
aws s3 ls s3://observatorio-inmobiliario/models/ --recursive
```

---

### Error: "Can't get attribute '__pyx_unpickle_CyHalfSquaredError'"

**Causa**: El modelo fue entrenado con una versión diferente de scikit-learn.

**Solución**:
1. **Opción A**: Re-entrenar el modelo con la versión de sklearn en `requirements.txt`
2. **Opción B**: Cambiar la versión en `requirements.txt` para que coincida con la del modelo

**Verificar versión usada en Lambda**:
```bash
# Ver logs de inicio - la versión se imprime al cargar
aws logs tail /aws/lambda/<FUNCTION_NAME> --region us-east-2 --since 5m | grep "sklearn"
```

---

### Error: "Image not found" al hacer deploy

**Causa**: La imagen no se pusheó correctamente a ECR, o se está usando el tag incorrecto.

**Solución**:
1. Verificar que la imagen existe en ECR:
```bash
export AWS_PROFILE=uade-valorar
aws ecr describe-images \
  --repository-name valorar-inference-ohe-py12 \
  --region us-east-2
```

2. Verificar el tag en `template.yaml` coincide con el tag en ECR

---

### Error: Docker build falla con arquitectura incorrecta

**Síntoma**: El build funciona localmente pero falla en Lambda.

**Causa**: Mac ARM/M1/M2 construye imágenes ARM por defecto, pero Lambda necesita x86_64.

**Solución**: Siempre usar `--platform linux/amd64`:
```bash
docker build --platform linux/amd64 -t my-image .
```

Todos los scripts `deploy-manual.sh` ya incluyen esto.

---

### Dos Lambdas apuntando al mismo ECR repo

**Problema**: Al deployar una Lambda nueva, "pisa" la imagen de otra Lambda existente.

**Síntomas**:
- Lambda A funciona bien
- Despliegas Lambda B
- Lambda A deja de funcionar
- Ambas usan el mismo ECR repository

**Solución**: Crear repositorios ECR separados para cada Lambda.

**Paso a paso**:

1. **Crear nuevo repositorio ECR**:
```bash
export AWS_PROFILE=uade-valorar
aws ecr create-repository \
  --repository-name valorar-inference-ohe-py12 \
  --region us-east-2
```

2. **Actualizar `samconfig.toml`**:
```toml
[default.deploy.parameters]
stack_name = "valorar-inference-py12"  # Nombre único
image_repositories = ["ohe=650532183679.dkr.ecr.us-east-2.amazonaws.com/valorar-inference-ohe-py12"]
```

3. **Actualizar `template.yaml`**:
```yaml
Properties:
  PackageType: Image
  ImageUri: 650532183679.dkr.ecr.us-east-2.amazonaws.com/valorar-inference-ohe-py12:latest
```

4. **Actualizar `deploy-manual.sh`**:
```bash
ECR_REPO="650532183679.dkr.ecr.us-east-2.amazonaws.com/valorar-inference-ohe-py12"
STACK_NAME="valorar-inference-py12"
```

5. **Restaurar la Lambda pisada** (si es necesario):
```bash
# Listar imágenes disponibles
aws ecr describe-images \
  --repository-name <OLD_REPO> \
  --region us-east-2

# Actualizar Lambda con imagen anterior
aws lambda update-function-code \
  --function-name <FUNCTION_NAME> \
  --image-uri <ECR_REPO>:<OLD_TAG> \
  --region us-east-2
```

---

### Ver todas las imágenes en un repositorio ECR

```bash
export AWS_PROFILE=uade-valorar
aws ecr describe-images \
  --repository-name valorar-inference-ohe-py12 \
  --region us-east-2 \
  --query 'sort_by(imageDetails, &imagePushedAt)[].[imageTags[0], imagePushedAt]' \
  --output table
```

---

### Error: "Update failed" en CloudFormation

**Síntoma**: El deploy falla y hace rollback.

**Debugging**:
1. Ver eventos del stack:
```bash
export AWS_PROFILE=uade-valorar
aws cloudformation describe-stack-events \
  --stack-name valorar-inference-py12 \
  --region us-east-2 \
  --max-items 20
```

2. Ver el mensaje de error específico en los eventos

3. Errores comunes:
   - Permisos IAM insuficientes
   - Imagen no existe en ECR
   - Timeout de Lambda muy bajo
   - Memoria insuficiente

---

## Estructura de Directorios

```
Machine Learning/
├── aws-docker-inference/          # Inference sin OHE
├── aws-docker-inference-ohe/      # Inference OHE Python 3.13 (prod)
├── aws-docker-inference-ohe-py12/ # Inference OHE Python 3.12 (nueva)
├── aws-docker-train/              # Training
└── aws-docker-report/             # Reporting

Cada directorio contiene:
├── Dockerfile                     # Definición de la imagen
├── lambda_function.py             # Código de la Lambda
├── requirements.txt               # Dependencias Python
├── template.yaml                  # SAM template (CloudFormation)
├── samconfig.toml                 # Configuración de SAM
├── build-locally.sh               # Build de Docker local
├── deploy-manual.sh               # Deploy completo automatizado
└── README.md                      # Documentación específica
```

---

## Convenciones de Naming

### Stacks CloudFormation
- Formato: `valorar-<purpose>[-variant]`
- Ejemplos:
  - `valorar-inference`
  - `valorar-inference-py12`
  - `valorar-train`

### ECR Repositories
- Formato: `valorar-<purpose>-<variant>`
- Ejemplos:
  - `valorar-inference-ohe-py12`
  - `valorar-train-models`

### Image Tags
- `latest`: Última versión (usar con cuidado)
- `<function>-<hash>-python<version>`: Tags específicos
  - Ejemplo: `ohe-4a70cb417aa9-python3.13`

### Lambda Functions
- SAM genera nombres automáticamente: `<stack>-<resource>-<random>`
- Ejemplo: `valorar-inference-py12-ohe-sUc6pJx4Qirq`

---

## Checklist para Nueva Lambda

- [ ] Crear directorio con estructura estándar
- [ ] Copiar y adaptar `Dockerfile`
- [ ] Escribir `lambda_function.py`
- [ ] Definir `requirements.txt`
- [ ] Crear `template.yaml` con ImageUri y Environment vars
- [ ] Crear nuevo ECR repository
- [ ] Configurar `samconfig.toml` con stack name único
- [ ] Crear `deploy-manual.sh` script
- [ ] Verificar arquitectura x86_64 en build
- [ ] Test local si es posible
- [ ] Deploy a AWS
- [ ] Verificar con invoke
- [ ] Verificar logs en CloudWatch
- [ ] Documentar en README específico

---

## AWS Profiles

El proyecto usa el profile `uade-valorar`:

```bash
export AWS_PROFILE=uade-valorar
```

Agregar a `.bashrc` o `.zshrc` para persistencia:
```bash
echo 'export AWS_PROFILE=uade-valorar' >> ~/.zshrc
```

---

## Recursos Útiles

- [AWS SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html)
- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [ECR Documentation](https://docs.aws.amazon.com/ecr/)
- [Docker Multi-platform Builds](https://docs.docker.com/build/building/multi-platform/)

---

## Mantenimiento

### Limpiar imágenes viejas en ECR

```bash
# Listar todas las imágenes
aws ecr describe-images \
  --repository-name <REPO_NAME> \
  --region us-east-2

# Eliminar imagen específica
aws ecr batch-delete-image \
  --repository-name <REPO_NAME> \
  --region us-east-2 \
  --image-ids imageTag=<TAG>
```

### Actualizar versión de scikit-learn

1. Cambiar en `requirements.txt`
2. Re-build imagen: `./deploy-manual.sh`
3. Re-entrenar modelos con la misma versión
4. Subir nuevos modelos a S3

---

## Contacto y Soporte

Para problemas específicos:
1. Revisar logs en CloudWatch
2. Verificar eventos de CloudFormation
3. Consultar este documento
4. Revisar README específico del directorio
