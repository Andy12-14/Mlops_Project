# MLOps Architecture Documentation

## Sentiment Analysis Pipeline

This document provides a comprehensive overview of our MLOps architecture for the sentiment analysis project, explaining technical choices and describing the complete workflow.

---

## 1. System Architecture Overview

```mermaid
flowchart TB
    subgraph Development["Development Environment"]
        DEV[Developer] --> CODE[Source Code]
        CODE --> GIT[Git Repository]
    end
    
    subgraph CI_CD["CI/CD Pipeline - GitHub Actions"]
        GIT --> TEST[test.yml]
        TEST -->|Pass| EVAL[evaluate.yml]
        EVAL -->|Pass| BUILD[build.yml]
    end
    
    subgraph Container["Docker Environment"]
        BUILD --> IMAGE[Docker Image]
        IMAGE --> REGISTRY[DockerHub Registry]
    end
    
    subgraph Production["Production Deployment"]
        REGISTRY --> CONTAINER[Docker Container]
        CONTAINER --> API[FastAPI Service]
        API --> USERS[Users/Applications]
    end
    
    subgraph Storage["Persistent Storage"]
        VOLUMES[(Docker Volumes)]
        VOLUMES --> MODEL[Trained Models]
        VOLUMES --> CACHE[HuggingFace Cache]
    end
    
    CONTAINER <--> VOLUMES
```

---

## 2. ML Pipeline Components

```mermaid
flowchart LR
    subgraph Data["Data Layer"]
        CSV[dataset.csv] --> EXTRACT[data_extraction.py]
        EXTRACT --> PROCESS[data_processing.py]
    end
    
    subgraph Processing["Processing Layer"]
        PROCESS --> CLEAN[Text Cleaning]
        CLEAN --> TOKEN[Tokenization]
        TOKEN --> SPLIT[Train/Val Split]
    end
    
    subgraph Model["Model Layer"]
        SPLIT --> TRAIN[model.py]
        TRAIN --> BERT[Fine-tuned BERT]
        BERT --> SAVE[model_outputs/]
    end
    
    subgraph Inference["Inference Layer"]
        SAVE --> LOAD[inference.py]
        LOAD --> PREDICT[Predictions]
    end
```

---

## 3. Technical Choices

### 3.1 Machine Learning Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Base Model** | BERT (bert-base-uncased) | Pre-trained transformer with strong NLP performance |
| **Framework** | PyTorch + HuggingFace Transformers | Industry standard, rich ecosystem |
| **Evaluation** | DistilBERT-SST2 | Fast evaluation with pre-trained sentiment model |
| **Serialization** | Joblib | Efficient serialization for scikit-learn compatible objects |

### 3.2 API & Serving

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Web Framework** | FastAPI | High performance, automatic OpenAPI docs, async support |
| **Server** | Uvicorn | ASGI server optimized for async Python |
| **Containerization** | Docker | Consistent environments, easy deployment |
| **Orchestration** | Docker Compose | Simple multi-service management |

### 3.3 CI/CD & DevOps

| Component | Technology | Justification |
|-----------|------------|---------------|
| **CI/CD** | GitHub Actions | Native GitHub integration, free for public repos |
| **Registry** | DockerHub | Industry standard container registry |
| **Linting** | Flake8 | Python style guide enforcement |
| **Testing** | Pytest | Feature-rich Python testing framework |

---

## 4. CI/CD Workflow

```mermaid
flowchart TD
    PUSH[Push to GitHub] --> TEST_TRIGGER{Triggered}
    
    subgraph test_yml["test.yml Workflow"]
        TEST_TRIGGER --> CHECKOUT1[Checkout Code]
        CHECKOUT1 --> SETUP1[Setup Python 3.11]
        SETUP1 --> DEPS1[Install Dependencies]
        DEPS1 --> LINT[Run Flake8 Linting]
        LINT --> PYTEST[Run Pytest]
    end
    
    PYTEST -->|Pass| EVAL_TRIGGER{Tests Passed}
    
    subgraph evaluate_yml["evaluate.yml Workflow"]
        EVAL_TRIGGER --> CHECKOUT2[Checkout Code]
        CHECKOUT2 --> SETUP2[Setup Python 3.11]
        SETUP2 --> DEPS2[Install Dependencies]
        DEPS2 --> EVALUATE[Run evaluate.py]
        EVALUATE --> CHECK{Accuracy >= 70%?}
        CHECK -->|Yes| UPLOAD[Upload Metrics Artifact]
        CHECK -->|No| FAIL[Workflow Fails]
    end
    
    UPLOAD --> BUILD_TRIGGER{Evaluation Passed}
    
    subgraph build_yml["build.yml Workflow"]
        BUILD_TRIGGER --> CHECKOUT3[Checkout Code]
        CHECKOUT3 --> BUILDX[Setup Docker Buildx]
        BUILDX --> LOGIN[Login to DockerHub]
        LOGIN --> BUILD[Build Docker Image]
        BUILD --> PUSH_IMG[Push to DockerHub]
    end
    
    PUSH_IMG --> DEPLOYED[Image Available on DockerHub]
```

---

## 5. Docker Architecture

### 5.1 Container Structure

```mermaid
flowchart TB
    subgraph Docker_Image["Docker Image (sentiment-api)"]
        BASE[Python 3.11-slim]
        BASE --> DEPS[Dependencies from requirements.txt]
        DEPS --> SRC[Source Code /app/src/]
        SRC --> DATA[Dataset /app/dataset/]
        DATA --> ENTRY[Entrypoint: uvicorn api:app]
    end
    
    subgraph Volumes["Docker Volumes"]
        V1[model_outputs] --> MODELS[Trained Models]
        V2[hf_cache] --> HF[HuggingFace Cache]
    end
    
    Docker_Image <--> Volumes
    
    ENTRY --> PORT[Expose Port 8000]
    PORT --> HEALTH[GET /health]
    PORT --> PREDICT[POST /predict]
```

### 5.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `PYTHONDONTWRITEBYTECODE=1` | Prevents .pyc file creation |
| `PYTHONUNBUFFERED=1` | Real-time log output |
| `PYTHONPATH=/app/src` | Module import paths |
| `TRANSFORMERS_CACHE=/app/.cache/huggingface` | HuggingFace model cache location |
| `HF_HOME=/app/.cache/huggingface` | HuggingFace home directory |
| `MODEL_PATH` | Path to trained model file |

---

## 6. API Endpoints

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant SentimentPredictor
    participant BERT Model
    
    Note over Client,BERT Model: Health Check Flow
    Client->>FastAPI: GET /health
    FastAPI-->>Client: {"status": "ok"}
    
    Note over Client,BERT Model: Prediction Flow
    Client->>FastAPI: POST /predict {"text": "..."}
    FastAPI->>SentimentPredictor: predict(text)
    SentimentPredictor->>SentimentPredictor: clean_text()
    SentimentPredictor->>BERT Model: tokenize & forward
    BERT Model-->>SentimentPredictor: logits
    SentimentPredictor->>SentimentPredictor: softmax & argmax
    SentimentPredictor-->>FastAPI: {"sentiment": "...", "confidence": ...}
    FastAPI-->>Client: JSON Response
```

---

## 7. Data Flow

```mermaid
flowchart LR
    subgraph Input
        RAW[Raw Reviews CSV]
    end
    
    subgraph Preprocessing
        RAW --> LOAD[load_data]
        LOAD --> CLEAN[clean_text]
        CLEAN --> LOWER[Lowercase]
        LOWER --> REMOVE[Remove URLs/Emails]
        REMOVE --> ALPHA[Keep Only Letters]
    end
    
    subgraph Tokenization
        ALPHA --> TOKENIZE[BERT Tokenizer]
        TOKENIZE --> IDS[input_ids]
        TOKENIZE --> MASK[attention_mask]
    end
    
    subgraph Inference
        IDS --> MODEL[BERT Model]
        MASK --> MODEL
        MODEL --> LOGITS[Logits]
        LOGITS --> SOFTMAX[Softmax]
        SOFTMAX --> PRED[Prediction]
    end
    
    subgraph Output
        PRED --> SENTIMENT[Positive/Negative]
        PRED --> CONF[Confidence Score]
        PRED --> PROBS[Class Probabilities]
    end
```

---

## 8. Project Files Summary

| Category | Files | Purpose |
|----------|-------|---------|
| **Source** | `src/data_extraction.py` | Load CSV data |
| | `src/data_processing.py` | Clean, tokenize, split data |
| | `src/model.py` | Train BERT classifier |
| | `src/inference.py` | Load model and predict |
| | `src/api.py` | FastAPI REST endpoints |
| | `src/evaluate.py` | Model evaluation script |
| **Tests** | `tests/unit/*.py` | Unit tests for all modules |
| **Docker** | `Dockerfile` | Container image definition |
| | `docker-compose.yml` | Service orchestration |
| | `.dockerignore` | Files excluded from build |
| **CI/CD** | `.github/workflows/test.yml` | Testing workflow |
| | `.github/workflows/evaluate.yml` | Model evaluation workflow |
| | `.github/workflows/build.yml` | Docker build workflow |
| **Config** | `requirements.txt` | Python dependencies |
| | `pytest.ini` | Pytest configuration |
| | `.flake8` | Linting configuration |

---

## 9. Deployment Instructions

### Local Development
```bash
# Setup
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# Run tests
pytest -q

# Run API locally
cd src && uvicorn api:app --reload
```

### Docker Deployment
```bash
# Build and run
docker compose up --build

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

### GitHub Actions Deployment
1. Add secrets to GitHub repository:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
2. Push to main branch
3. Workflows automatically trigger
4. Image published to DockerHub

---

## 10. Quality Gates

| Stage | Gate | Threshold |
|-------|------|-----------|
| Linting | Flake8 | No critical errors (E9, F63, F7, F82) |
| Unit Tests | Pytest | All tests pass |
| Model Evaluation | Accuracy | ≥ 70% |
| Docker Build | Build Success | Image builds without errors |

> Pipeline fails if any gate is not met, preventing broken code from being deployed.
