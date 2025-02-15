# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.13

# Copy application files to the container
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
COPY alquilerescaba_202501.xlsx ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Lambda function
CMD ["lambda_function.lambda_handler"]
