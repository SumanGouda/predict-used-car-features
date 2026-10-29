# Mlflow Run command 
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui

# Docker container run 
docker run -p 8000:8000 sumangouda/car-analytics:latest

# uvicorn run command 
uvicorn app:app --reload
