# Optional custom Dockerfile.
# The recommended approach (see README.md) is to let MLflow build the image
# for you directly from the Model Registry with:
#     mlflow models build-docker --model-uri "models:/ChurnModel/Production" --name churn-model-image
#
# This Dockerfile is a fallback if you'd rather build it yourself around
# the mlflow pyfunc CLI (e.g. to add custom system deps).

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow

# Copy the MLflow model directory you exported with:
#   mlflow artifacts download -u "models:/ChurnModel/Production" -d ./exported_model
COPY exported_model /app/model

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]