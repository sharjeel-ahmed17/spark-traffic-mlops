# 🚦 Spark Traffic MLOps

A complete end-to-end MLOps pipeline for traffic vehicle prediction using Apache Spark MLlib, MLflow, DVC, and modern deployment practices.

[![CI/CD Pipeline](https://github.com/sharjeel-ahmed17/spark-traffic-mlops/actions/workflows/deploy.yml/badge.svg)](https://github.com/sharjeel-ahmed17/spark-traffic-mlops/actions)
[![Docker Image](https://img.shields.io/docker/v/sharjeelahmed017/traffic-prediction-api?label=Docker)](https://hub.docker.com/r/sharjeelahmed017/traffic-prediction-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [DVC Pipeline](#dvc-pipeline)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Live Demos](#live-demos)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a production-ready MLOps pipeline for predicting traffic vehicle counts at different junctions. It leverages Apache Spark for distributed data processing and machine learning, MLflow for experiment tracking, DVC for data versioning, and includes complete CI/CD automation with Docker and Kubernetes deployment configurations.

## ✨ Features

- **Distributed ML Pipeline**: Apache Spark MLlib for scalable machine learning
- **Experiment Tracking**: MLflow integration with DagsHub for experiment management
- **Data Versioning**: DVC for reproducible data pipelines
- **Model Registry**: MLflow Model Registry with production/staging environments
- **CI/CD Automation**: GitHub Actions workflow for automated training and deployment
- **Containerization**: Docker images with multi-stage builds
- **Orchestration**: Kubernetes manifests for production deployment
- **REST API**: FastAPI service for real-time predictions
- **Interactive UI**: Streamlit dashboard for model interaction
- **Multiple Deployments**: Hugging Face Spaces, Vercel, and Streamlit Cloud

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Source    │
│  (Kaggle)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DVC Pipeline   │
│  ├─ Extraction  │
│  ├─ Cleaning    │
│  ├─ Transform   │
│  └─ Training    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow         │
│  Tracking       │
│  (DagsHub)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Registry │
│  (Production)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │
│  Service        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deployment     │
│  ├─ Docker      │
│  ├─ K8s         │
│  └─ HF Spaces   │
└─────────────────┘
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11**: Primary programming language
- **Apache Spark 4.1.1**: Distributed data processing and ML
- **PySpark**: Python API for Spark
- **MLflow 3.12.0**: Experiment tracking and model registry
- **DVC 3.67.1**: Data version control
- **FastAPI 0.136.1**: REST API framework
- **Streamlit 1.57.0**: Interactive web UI

### ML & Data Science
- **scikit-learn 1.8.0**: Additional ML utilities
- **pandas 2.3.3**: Data manipulation
- **numpy 2.4.4**: Numerical computing

### DevOps & Deployment
- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **GitHub Actions**: CI/CD automation
- **DagsHub**: MLflow and DVC remote storage
- **Hugging Face Spaces**: Model deployment

### Models Implemented
- Poisson Generalized Linear Model (GLM)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosted Trees

## 📁 Project Structure

```
spark-traffic-mlops/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline configuration
├── app/
│   └── main.py                 # FastAPI application
├── data/
│   ├── raw/                    # Raw data from Kaggle
│   ├── processed/              # Cleaned data
│   └── transformed/            # Train/test splits
├── docs/
│   ├── documentation.md        # Detailed documentation
│   └── server.md               # Server setup guide
├── k8s/                        # Kubernetes manifests
│   ├── deployment.yaml
│   ├── services.yaml
│   ├── configmap.yaml
│   └── pvc.yaml
├── models/
│   └── pipeline/               # Trained pipeline model
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── src/
│   ├── extraction.py           # Data extraction from Kaggle
│   ├── cleaning.py             # Data cleaning pipeline
│   ├── transformation.py       # Feature engineering
│   ├── train.py                # Model training with MLflow
│   ├── best_model.py           # Best model selection
│   ├── load.py                 # Model loading utilities
│   ├── logger.py               # Logging configuration
│   └── config_utils.py         # Configuration utilities
├── ui/
│   ├── app.py                  # Streamlit UI (full version)
│   └── streamlit_app.py        # Streamlit UI (minimal)
├── config.yaml                 # Project configuration
├── params.yaml                 # Pipeline parameters
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock file
├── docker-compose.yaml         # Multi-container setup
├── Dockerfile                  # Container image definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Java 17+ (for Spark)
- Git
- Docker (optional)
- Kaggle API credentials

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/sharjeel-ahmed17/spark-traffic-mlops.git
cd spark-traffic-mlops
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/sharjeel-ahmed17/spark-traffic-mlops.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token
DAGSHUB_USERNAME=your_username
DAGSHUB_PASSWORD=your_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

5. **Configure DVC remote**

```bash
dvc remote add -d dagshub https://dagshub.com/sharjeel-ahmed17/spark-traffic-mlops.dvc
dvc remote modify dagshub auth basic
dvc remote modify dagshub user $DAGSHUB_USERNAME
dvc remote modify dagshub password $DAGSHUB_PASSWORD
```

### Quick Start

1. **Pull existing data and models**

```bash
dvc pull
```

2. **Run the complete pipeline**

```bash
dvc repro
```

3. **Start the API server**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. **Launch Streamlit UI**

```bash
streamlit run ui/streamlit_app.py
```

## 🔄 DVC Pipeline

The project uses DVC to manage a reproducible ML pipeline with four stages:

### Pipeline Stages

#### 1. Extraction
Downloads traffic dataset from Kaggle and saves to `data/raw/`

```bash
dvc repro extraction
```

**Dependencies**: `src/extraction.py`  
**Outputs**: `data/raw/traffic.csv`

#### 2. Cleaning
Cleans and validates the raw data

```bash
dvc repro cleaning
```

**Dependencies**: `src/cleaning.py`, `data/raw/traffic.csv`  
**Outputs**: `data/processed/clean_traffic_data`

#### 3. Transformation
Performs feature engineering and train/test split

```bash
dvc repro transformation
```

**Dependencies**: `src/transformation.py`, `data/processed/clean_traffic_data`  
**Outputs**: 
- `data/transformed/train`
- `data/transformed/test`
- `models/pipeline`

#### 4. Training
Trains multiple models with hyperparameter tuning and logs to MLflow

```bash
dvc repro train
```

**Dependencies**: `src/train.py`, transformed data  
**Outputs**: `scores.json`

### View Pipeline DAG

```bash
dvc dag
```

### Pipeline Parameters

Edit `params.yaml` to customize:

- Dataset configuration
- Feature engineering parameters
- Model hyperparameters
- Cross-validation settings

## 🎓 Model Training

### Supported Models

The pipeline trains and compares four regression models:

1. **Poisson GLM**: For count data with Poisson distribution
2. **Decision Tree**: Non-linear relationships
3. **Random Forest**: Ensemble of decision trees
4. **Gradient Boosting**: Sequential ensemble learning

### Hyperparameter Tuning

Cross-validation with grid search is performed for each model. Configure in `params.yaml`:

```yaml
training:
  cv_folds: 3
  seed: 42
  models:
    PoissonGLM:
      regParam: [0.001, 0.01, 0.1]
      maxIter: [25, 50]
    DecisionTree:
      maxDepth: [3, 5, 7]
      minInstancesPerNode: [1, 5, 10]
    RandomForest:
      numTrees: [50, 100]
      maxDepth: [5, 10]
    GradientBoosting:
      maxDepth: [3, 5]
      maxIter: [20, 50]
```

### Experiment Tracking

All experiments are logged to MLflow with:
- Model parameters
- Training metrics (RMSE, MAE, R²)
- Model artifacts
- Feature importance

View experiments at: [DagsHub MLflow UI](https://dagshub.com/sharjeel-ahmed17/spark-traffic-mlops.mlflow/)

## 🐳 Deployment

### Docker

#### Build and run locally

```bash
docker build -t traffic-prediction-api .
docker run -p 8000:8000 traffic-prediction-api
```

#### Pull from Docker Hub

```bash
docker pull sharjeelahmed017/traffic-prediction-api:latest
docker run -p 8000:8000 sharjeelahmed017/traffic-prediction-api:latest
```

#### Docker Compose

```bash
docker-compose up -d
```

This starts:
- FastAPI service on port 8000
- Streamlit UI on port 8501

### Kubernetes

Deploy to Kubernetes cluster:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/
```

Check deployment status:

```bash
kubectl get pods -n traffic-mlops
kubectl get services -n traffic-mlops
```

### CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **On Push/PR**: 
   - Runs DVC pipeline
   - Trains models with MLflow
   - Pushes data to DagsHub
   - Updates `dvc.lock`

2. **On Main Branch Push**:
   - Builds Docker image
   - Pushes to Docker Hub
   - Triggers Hugging Face Space restart

## 📡 API Documentation

### Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "message": "Traffic Vehicle Prediction API Running"
}
```

#### Predict
```http
POST /predict
```

**Request Body:**
```json
{
  "junction": 1,
  "day_of_week": 3,
  "hour": 14,
  "month": 5,
  "year": 2024
}
```

**Response:**
```json
{
  "predicted_vehicles": 42.5,
  "model_name": "Traffic_Vehicle_Prediction",
  "status": "success"
}
```

### Interactive API Docs

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🌐 Live Demos

### Production Deployments

| Service | URL | Description |
|---------|-----|-------------|
| **FastAPI** | [sharjeel17-traffic.hf.space](https://sharjeel17-traffic.hf.space/) | REST API endpoint |
| **Streamlit MVP** | [traffic-vehicle-prediction.streamlit.app](https://traffic-vehicle-prediction.streamlit.app/) | Interactive dashboard |
| **React UI** | [traffic-prediction2.vercel.app](https://traffic-prediction2.vercel.app/dashboard) | Modern web interface |
| **HF Insights** | [sharjeel17-traffic-insights.hf.space](https://sharjeel17-traffic-insights.hf.space/) | AWS deployment |
| **MLflow UI** | [dagshub.com/.../mlflow](https://dagshub.com/sharjeel-ahmed17/spark-traffic-mlops.mlflow/) | Experiment tracking |
| **DVC Remote** | [dagshub.com/.../dvc](https://dagshub.com/sharjeel-ahmed17/spark-traffic-mlops.dvc/) | Data versioning |
| **GitHub Repo** | [github.com/sharjeel-ahmed17/spark-traffic-mlops](https://github.com/sharjeel-ahmed17/spark-traffic-mlops) | Source code |
| **DagsHub Repo** | [dagshub.com/sharjeel-ahmed17/spark-traffic-mlops](https://dagshub.com/sharjeel-ahmed17/spark-traffic-mlops) | MLOps platform |

### Docker Images

```bash
# Latest version
docker pull sharjeelahmed017/traffic-prediction-api:latest

# Specific commit
docker pull sharjeelahmed017/traffic-prediction-api:4632c0542d0f5ab0ea2a1b669c6b1d4a67b7dd41
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**

2. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add tests if applicable
   - Update documentation

4. **Commit your changes**
```bash
git add .
git commit -m "feat: add your feature description"
```

5. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Ensure CI/CD checks pass

### Development Guidelines

- Use meaningful commit messages (conventional commits)
- Write docstrings for functions and classes
- Keep functions small and focused
- Add type hints where appropriate
- Update tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sharjeel Ahmed**

- GitHub: [@sharjeel-ahmed17](https://github.com/sharjeel-ahmed17)
- DagsHub: [@sharjeel-ahmed17](https://dagshub.com/sharjeel-ahmed17)

## 🙏 Acknowledgments

- Dataset: [Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset) by fedesoriano
- MLflow tracking powered by [DagsHub](https://dagshub.com)
- Deployment on [Hugging Face Spaces](https://huggingface.co/spaces)

## 📊 Project Stats

- **Models Trained**: 4 (Poisson GLM, Decision Tree, Random Forest, GBT)
- **Pipeline Stages**: 4 (Extraction, Cleaning, Transformation, Training)
- **Deployment Platforms**: 5 (Docker Hub, HF Spaces, Streamlit Cloud, Vercel, K8s)
- **CI/CD**: Fully automated with GitHub Actions

---

**Built with ❤️ using Apache Spark, MLflow, and DVC**
