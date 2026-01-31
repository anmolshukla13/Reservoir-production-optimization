# Reservoir Production Optimization - Deployment Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Dataset Setup](#dataset-setup)
4. [Model Training](#model-training)
5. [API Deployment](#api-deployment)
6. [Docker Deployment](#docker-deployment)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Cloud Deployment (AWS/Azure/GCP)](#cloud-deployment)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 20GB free space
- **CPU**: 4+ cores recommended

### Software Requirements
```bash
# Required
- Python 3.9+
- Docker 20.10+
- Docker Compose 2.0+
- Git

# Optional (for Kubernetes)
- kubectl 1.24+
- Kubernetes cluster (minikube, kind, or cloud provider)
- Helm 3.0+
```

---

## 🚀 Local Development Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/reservoir-production-optimization.git
cd reservoir-production-optimization
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Environment Configuration

```bash
# Create .env file
cat > .env << EOF
ENVIRONMENT=development
LOG_LEVEL=debug
DATABASE_URL=postgresql://postgres:password@localhost:5432/reservoir_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-change-this
API_VERSION=1.0.0
EOF
```

---

## 📊 Dataset Setup

### Option 1: Use Synthetic Data (Recommended for Quick Start)

```bash
# Generate synthetic dataset
python data_generator.py

# Output:
# - data/synthetic/well_properties.csv
# - data/synthetic/production_data.csv
# - data/synthetic/full_dataset.csv
```

### Option 2: Use Volve Field Data (Real Data)

#### Download Steps:

1. **Visit**: https://www.equinor.com/energy/volve-data-sharing
2. **Register**: Create free account
3. **Download**: Production data package (~500MB)
4. **Extract**: Place in `data/raw/` directory

```bash
# Create data directories
mkdir -p data/raw data/processed

# Move downloaded files
mv ~/Downloads/volve_production_data.csv data/raw/

# Verify data
python -c "
import pandas as pd
df = pd.read_csv('data/raw/volve_production_data.csv')
print(f'Loaded {len(df):,} records')
print(df.head())
"
```

### Option 3: Use Other Public Datasets

**NLOG (Netherlands):**
```bash
# Download Dutch oil & gas data
wget https://www.nlog.nl/en/data/production_data.csv -O data/raw/nlog_production.csv
```

**Kansas Geological Survey:**
```bash
# Download Kansas production data
wget http://www.kgs.ku.edu/PRS/data/production.csv -O data/raw/kansas_production.csv
```

---

## 🎓 Model Training

### Step 1: Data Preprocessing

```bash
python preprocessing.py

# This will:
# - Load raw data
# - Handle missing values
# - Create engineered features
# - Split train/test sets
# - Save processed data
```

### Step 2: Train Models

```bash
# Train all models
python model_training.py

# Output:
# - Trained models in models/
# - Model comparison report
# - Performance metrics
```

### Step 3: Hyperparameter Tuning (Optional)

```bash
# Tune specific model
python -c "
from model_training import ReservoirMLModels
from preprocessing import preprocess_pipeline

X_train, X_test, y_train, y_test, _ = preprocess_pipeline(
    'data/synthetic/full_dataset.csv', 
    target='oil_rate'
)

trainer = ReservoirMLModels()
trainer.initialize_models()
best_model, params = trainer.tune_hyperparameters(
    'XGBoost', X_train, y_train, method='random'
)
print(f'Best parameters: {params}')
"
```

### Step 4: Evaluate Models

```bash
# Generate evaluation report
python -c "
from model_training import ReservoirMLModels
import pandas as pd

trainer = ReservoirMLModels()
# Load results
report = pd.read_csv('models/model_comparison_report.csv')
print(report)
"
```

---

## 🔌 API Deployment

### Local API Server

```bash
# Start API server
uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at:
# http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "well_properties": {
      "porosity": 0.22,
      "permeability": 150.0,
      "net_pay": 25.0,
      "initial_pressure": 3500.0,
      "reservoir_temperature": 180.0,
      "measured_depth": 8500.0,
      "true_vertical_depth": 7800.0,
      "skin_factor": 2.0,
      "tubing_diameter": 3.5,
      "choke_size": 32,
      "oil_api": 35.0,
      "gas_gravity": 0.65
    },
    "production_data": {
      "days_on_production": 365,
      "oil_rate": 500.0,
      "gas_rate": 5000.0,
      "water_rate": 200.0,
      "reservoir_pressure": 3200.0,
      "wellhead_pressure": 800.0,
      "water_cut": 30.0,
      "gor": 1000.0
    }
  }'
```

### Start Dashboard

```bash
# In a new terminal
streamlit run dashboard.py

# Dashboard will be available at:
# http://localhost:8501
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build API image
docker build -t reservoir-api:latest .

# Build dashboard image
docker build -t reservoir-dashboard:latest -f Dockerfile.dashboard .
```

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Individual Container Management

```bash
# Run API only
docker run -d -p 8000:8000 --name reservoir-api reservoir-api:latest

# Run with volume mounting
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name reservoir-api reservoir-api:latest

# Check container status
docker ps

# View logs
docker logs -f reservoir-api

# Stop container
docker stop reservoir-api

# Remove container
docker rm reservoir-api
```

---

## ☸️ Kubernetes Deployment

### Prerequisites Setup

```bash
# Install kubectl (if not installed)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# For local testing, install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start minikube
minikube start --cpus=4 --memory=8192
```

### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f kubernetes-deployment.yaml

# Check deployment status
kubectl get deployments -n reservoir-prod
kubectl get pods -n reservoir-prod
kubectl get services -n reservoir-prod

# View logs
kubectl logs -f deployment/reservoir-api -n reservoir-prod

# Scale deployment
kubectl scale deployment reservoir-api --replicas=5 -n reservoir-prod

# Update deployment (rolling update)
kubectl set image deployment/reservoir-api api=reservoir-api:v2.0 -n reservoir-prod

# Rollback deployment
kubectl rollout undo deployment/reservoir-api -n reservoir-prod
```

### Expose Services

```bash
# For minikube
minikube service reservoir-api-service -n reservoir-prod
minikube service reservoir-dashboard-service -n reservoir-prod

# Get service URLs
kubectl get svc -n reservoir-prod
```

### Configure Ingress

```bash
# Enable ingress on minikube
minikube addons enable ingress

# Apply ingress
kubectl apply -f kubernetes-deployment.yaml

# Get ingress IP
kubectl get ingress -n reservoir-prod
```

---

## ☁️ Cloud Deployment

### AWS Deployment (EKS)

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS
aws configure

# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name reservoir-cluster \
  --region us-east-1 \
  --nodegroup-name reservoir-nodes \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name reservoir-cluster

# Deploy application
kubectl apply -f kubernetes-deployment.yaml

# Setup load balancer
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.0/deploy/static/provider/aws/deploy.yaml
```

### Azure Deployment (AKS)

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create resource group
az group create --name reservoir-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group reservoir-rg \
  --name reservoir-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 5

# Get credentials
az aks get-credentials --resource-group reservoir-rg --name reservoir-cluster

# Deploy application
kubectl apply -f kubernetes-deployment.yaml
```

### GCP Deployment (GKE)

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Create GKE cluster
gcloud container clusters create reservoir-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5

# Get credentials
gcloud container clusters get-credentials reservoir-cluster --zone us-central1-a

# Deploy application
kubectl apply -f kubernetes-deployment.yaml
```

---

## 📊 Monitoring & Logging

### Prometheus Setup

```bash
# Access Prometheus
kubectl port-forward -n reservoir-prod svc/prometheus 9090:9090

# Visit: http://localhost:9090
```

### Grafana Dashboard

```bash
# Access Grafana
kubectl port-forward -n reservoir-prod svc/grafana 3000:3000

# Visit: http://localhost:3000
# Login: admin / admin
```

### MLflow Tracking

```bash
# Access MLflow
kubectl port-forward -n reservoir-prod svc/mlflow 5000:5000

# Visit: http://localhost:5000
```

### Application Logs

```bash
# View API logs
kubectl logs -f deployment/reservoir-api -n reservoir-prod --tail=100

# View all pod logs
kubectl logs -f --all-containers=true -l app=reservoir-api -n reservoir-prod

# Export logs to file
kubectl logs deployment/reservoir-api -n reservoir-prod > api-logs.txt
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

#### Issue: Docker Build Fails

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t reservoir-api:latest .
```

#### Issue: Kubernetes Pods Not Starting

```bash
# Describe pod for errors
kubectl describe pod <pod-name> -n reservoir-prod

# Check events
kubectl get events -n reservoir-prod --sort-by='.lastTimestamp'

# Check resource limits
kubectl top nodes
kubectl top pods -n reservoir-prod
```

#### Issue: Model Not Loading

```bash
# Check model file exists
ls -lh models/

# Verify model file
python -c "
import joblib
model = joblib.load('models/xgboost_model.pkl')
print('Model loaded successfully!')
"
```

### Performance Optimization

```bash
# Enable GPU (if available)
docker run --gpus all -d reservoir-api:latest

# Increase worker processes
uvicorn api_main:app --workers 4

# Optimize database queries
# Add indexes in PostgreSQL
CREATE INDEX idx_well_id ON production_data(well_id);
CREATE INDEX idx_date ON production_data(date);
```

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: 
- **Email**: anmolshukla505@gmail.com

---

## 📝 Next Steps

1. ✅ Complete local setup
2. ✅ Train models with your data
3. ✅ Test API endpoints
4. ✅ Deploy to staging environment
5. ✅ Run performance tests
6. ✅ Deploy to production
7. ✅ Setup monitoring
8. ✅ Configure alerts

---

**Happy Deploying! 🚀**
