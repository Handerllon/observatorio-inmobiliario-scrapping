# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.13

# Copy application files to the container
# Install system dependencies
RUN yum install -y wget unzip

# Install ChromeDriver
RUN wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip && \
    mv chromedriver /usr/bin/

# Install Headless Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm && \
    yum install -y ./google-chrome-stable_current_x86_64.rpm

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Working Directory
COPY ../orchestration/ /

# Command to run the Lambda function
CMD ["init_lambda.lambda_handler"]
