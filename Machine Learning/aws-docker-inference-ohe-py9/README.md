# Lambda Inference OHE - Python 3.9

Lambda function para inferencia de precios de alquileres usando scikit-learn con One-Hot Encoding.

## Arquitectura

- **Runtime**: Python 3.9 (Container Image)
- **Arquitectura**: x86_64
- **Memoria**: 256 MB
- **Timeout**: 60 segundos

## Build y Deploy

### Problema conocido: SAM no detecta Docker en Mac

SAM CLI tiene problemas detectando Docker en macOS. Por eso usamos build manual con Docker:

```bash
# 1. Build de la imagen (forzando arquitectura x86_64)
./build-locally.sh

# 2. Login a ECR (usa tu AWS account id)
export AWS_PROFILE=uade-valorar
export ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com"

# 3. Tag de la imagen
docker tag lambda-docker-image-py9:latest "$ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com/valorar-inference-ohe-py9:latest"

# 4. Push a ECR
docker push "$ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com/valorar-inference-ohe-py9:latest"

# 5. Deploy con SAM
cd "Machine Learning/aws-docker-inference-ohe-py9"
export AWS_PROFILE=uade-valorar
sam deploy --stack-name valorar-inference-py9 --region us-east-2 --capabilities CAPABILITY_IAM --resolve-s3
```

### Cambios importantes en los archivos

1. **Dockerfile**: Base image `public.ecr.aws/lambda/python:3.9`
2. **build-locally.sh**: Agregado `--platform linux/amd64` para forzar x86_64 en Mac ARM
3. **lambda_function.py**: Agregada validación para evitar `IndexError` cuando no hay modelos en S3
4. **Tags/stack**: Renombrados a `py9` (ECR repo, stack name, imagen local)
5. **Account ID**: Se obtiene en runtime con `aws sts get-caller-identity` en vez de estar hardcodeado

## Testing

### Listar Lambda functions

```bash
export AWS_PROFILE=uade-valorar
aws lambda list-functions --region us-east-2 --query "Functions[?contains(FunctionName, 'valorar')].[FunctionName, LastModified]" --output table
```

### Ver detalles de la función

```bash
export AWS_PROFILE=uade-valorar
aws lambda get-function --function-name valorar-inference-ohe-PX2BhKiczrDk --region us-east-2 --query '{Name:Configuration.FunctionName, Memory:Configuration.MemorySize, Timeout:Configuration.Timeout, Image:Code.ImageUri}' --output json
```

### Invocar la Lambda desde command line

```bash
export AWS_PROFILE=uade-valorar
aws lambda invoke \
  --function-name valorar-inference-ohe-PX2BhKiczrDk \
  --region us-east-2 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"total_area": 50, "rooms": 2, "bedrooms": 1, "antiquity": 10, "neighborhood": "PALERMO"}' \
  /tmp/response.json && cat /tmp/response.json
```

### Ver logs en CloudWatch

```bash
export AWS_PROFILE=uade-valorar
aws logs tail /aws/lambda/valorar-inference-ohe-PX2BhKiczrDk --region us-east-2 --follow
```

## Payload de ejemplo

```json
{
  "total_area": 50,
  "rooms": 2,
  "bedrooms": 1,
  "antiquity": 10,
  "neighborhood": "PALERMO"
}
```

**Neighborhoods soportados:**
- ALMAGRO, BALVANERA, BELGRANO, CABALLITO, COLEGIALES, DEVOTO
- FLORES, MONTSERRAT, NUNEZ, PALERMO, PARQUE PATRICIOS, PUERTO MADERO
- RECOLETA, RETIRO, SAN NICOLAS, SAN TELMO, VILLA CRESPO, VILLA DEL PARQUE
- VILLA URQUIZA

## Variables de entorno

- `BUCKET_NAME`: `observatorio-inmobiliario` - Bucket S3 donde se almacenan los modelos

## Selección automática de modelo

La Lambda busca automáticamente el modelo más reciente en S3:
- Busca archivos en la carpeta `models/` con extensión `.joblib`
- Excluye archivos que contengan `_wo_` o `by-neighborhood` en el nombre
- Ordena por fecha (formato `DDMMYYYY` en el nombre del archivo)
- Selecciona el más reciente

## Troubleshooting

### Error: "IndexError: list index out of range"
- **Causa**: No hay modelos en el bucket S3 que cumplan los criterios
- **Solución**: Verificar que existan archivos `.joblib` en `s3://observatorio-inmobiliario/models/`

### Error: SAM no detecta Docker
- **Causa**: SAM CLI no encuentra el socket de Docker en macOS
- **Solución**: Usar el proceso manual de build + push a ECR descrito arriba

### Error: Incompatibilidad de versión de scikit-learn
- **Causa**: El modelo fue entrenado con una versión diferente de scikit-learn
- **Solución**: Re-entrenar el modelo con la misma versión especificada en `requirements.txt`
