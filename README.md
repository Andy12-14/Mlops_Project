# MLOps Project — Sentiment Analysis of User Reviews

This repository contains a small MLOps-style pipeline for performing sentiment analysis on user reviews. It includes data extraction, preprocessing, a model training pipeline (fine-tuning a transformer), and an inference interface with unit tests.

This README explains how to set up the environment, run the pipeline, and use the provided modules.

## Repo layout

```
Mlops_Project/
├── .github/
│   └── workflows/
│       ├── test.yml              # CI: Runs tests and linting on push/PR
│       ├── evaluate.yml          # CI: Model evaluation with thresholds
│       └── build.yml             # CD: Build and publish Docker image
├── dataset/
│   └── dataset.csv               # Raw dataset (CSV)
├── src/
│   ├── api.py                    # FastAPI REST API for predictions
│   ├── data_extraction.py        # load_data(...) - reads dataset.csv
│   ├── data_processing.py        # process_dataframe(...) - cleaning, tokenization, split
│   ├── model.py                  # SentimentClassifier (train/evaluate/save)
│   └── inference.py              # SentimentPredictor (load model and predict)
├── tests/
│   └── unit/
│       ├── test_model.py         # Tests for SentimentClassifier
│       ├── test_inference.py     # Tests for SentimentPredictor
│       ├── test_data_extraction.py
│       └── test_data_processing.py
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Multi-service orchestration
├── .dockerignore                 # Files excluded from Docker build
├── requirements.txt              # Python dependencies
├── pytest.ini
└── README.md
```

Brief component descriptions
- `dataset/dataset.csv`: CSV of user reviews and any metadata. Keep sensitive data out of the repo.
- `src/data_extraction.py`: Contains helpers to load the CSV into a pandas DataFrame (e.g., `load_data(path)`).
- `src/data_processing.py`: Preprocessing pipeline functions such as `process_dataframe(df, ...)` that clean text, tokenize (when needed), label or map sentiment targets, and split into train/validation sets.
- `src/model.py`: Defines the model training and evaluation. Main class is expected as `SentimentClassifier` with methods to `train(...)`, `evaluate(...)`, and `save(output_dir)`.
- `src/inference.py`: Lightweight predictor `SentimentPredictor` which loads a saved model and exposes `predict(text_or_list)` returning labels and confidence scores.
- `tests/unit/*`: Pytest unit tests that exercise the above modules.

## Quick setup (Windows with bash)

The instructions below assume you are running the `bash.exe` shell (Git Bash / WSL-compatible). Adjust the virtual environment commands for plain PowerShell or CMD if needed.

1. Create and activate a virtual environment

```bash
cd "c:/Users/oman/Desktop/MLops project/Mlops_Project"
python -m venv .venv
source .venv/Scripts/activate  # on Git Bash this works; on WSL use source .venv/bin/activate
```

2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer, create an editable install for development:

```bash
pip install -e .
```

Note: `requirements.txt` should list packages like pandas, pytest and transformers if used by the code. If anything is missing, add it and re-run `pip install -r requirements.txt`.

## Usage examples

Below are common usage patterns. The exact function signatures in your `src` modules may vary slightly — these examples show the intended contract.

1) Load data (data_extraction)

```python
from src.data_extraction import load_data

df = load_data("dataset/dataset.csv")
print(df.head())
```

2) Preprocess and split (data_processing)

```python
from src.data_processing import process_dataframe

# process_dataframe should return at least: train_df, val_df
train_df, val_df = process_dataframe(df, text_col="review_text", label_col="label")
```

3) Train model (model)

```python
from src.model import SentimentClassifier

clf = SentimentClassifier(model_name="bert-base-uncased")
clf.train(train_df, val_df, epochs=2, batch_size=16, output_dir="model_outputs")
clf.save("model_outputs/final")
```

4) Inference (inference)

```python
from src.inference import SentimentPredictor

predictor = SentimentPredictor("model_outputs/final")
print(predictor.predict("I love this product!"))
print(predictor.predict(["Great app", "It crashed on start"]))
```

If your code expects CLI entrypoints, you can also run each module directly:

```bash
python src/data_processing.py
python src/model.py --epochs 3 --batch-size 16 --output-dir model_outputs
python src/inference.py --text "This is great"
```

## Containerization (Docker)

- Build the API image from the project root:

```bash
docker build -t sentiment-api .
```

- Start the stack (serves FastAPI on port 8000):

```bash
docker compose up --build
```

Services and volumes:
- `api` (FastAPI) on `8000`.
- Volumes: `model_outputs` for the trained `sentiment_classifier.joblib`, `hf_cache` for Hugging Face cache. `dataset` is mounted read-only into the container.

If you train locally, place the exported model in `model_outputs/` so the container can load it.

## FastAPI endpoints

- `GET /health` - health check
- `POST /predict` - body: `{"text": "...", "clean_texts": true}`; returns sentiment, confidence, and class probabilities.

## Tests

Run unit tests with pytest from the project root:

```bash
cd "c:/Users/oman/Desktop/MLops project/Mlops_Project"
pytest -q
```

If you want a single test file run:

```bash
pytest -q tests/unit/test_model.py
```

## CI/CD (GitHub Actions)

- `.github/workflows/test.yml`: runs lint (flake8) and pytest on every push/PR.
- `.github/workflows/evaluate.yml`: runs after Tests succeed (or manually), evaluates the model with `src/evaluate.py`, uploads `metrics/metrics.json`, and fails if accuracy is below the threshold.
- `.github/workflows/build.yml`: builds the Docker image and pushes to DockerHub using `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets (push skipped on PRs).

## Small contract & expectations

- Inputs: CSV file in `dataset/` with a text column (e.g., `review_text`) and optionally `label`.
- Outputs: Trained model artifacts saved to `model_outputs/` and evaluation metrics printed/logged during training.
- Errors: Modules should raise informative exceptions when files are missing or input shapes are incorrect. Functions should validate required columns.

Edge cases to consider
- Empty or missing CSV file — fail fast with a clear message.
- Very short or non-English text — preprocessing should handle or drop these safely.
- Large datasets — consider streaming or batching.

## Development tips

- Add missing packages to `requirements.txt` and pin versions for reproducibility.
- Add a `Makefile` or small CLI wrapper if you repeatedly run the same sequence of steps.
- For CI, run `pytest` and a lightweight linting step (e.g., flake8).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new behaviour
4. Open a pull request

---

## Docker Containerization

### Dockerfile

The `Dockerfile` creates a containerized environment for the sentiment analysis API:
- **Base image**: Python 3.11-slim
- **Dependencies**: Installs all requirements from `requirements.txt`
- **Entry point**: FastAPI server on port 8000

### Docker Compose

The `docker-compose.yml` orchestrates the services:

| Service | Description |
|---------|-------------|
| `api` | FastAPI sentiment prediction service on port 8000 |

**Volumes**:
- `model_outputs` - Persists trained models between container restarts
- `hf_cache` - Caches HuggingFace model downloads

### Running with Docker

```bash
# Build and start the container
docker compose up --build

# Test the API
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Stop the container
docker compose down
```

---

## REST API

The `src/api.py` module provides a FastAPI-based REST interface:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/predict` | POST | Predict sentiment for a single text |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!", "clean_texts": true}'
```

### Example Response

```json
{
  "text": "this movie was fantastic",
  "sentiment": "Positive",
  "confidence": 0.98,
  "probabilities": {"negative": 0.02, "positive": 0.98}
}
```

---

## CI/CD with GitHub Actions

Three workflows automate testing, evaluation, and deployment:

### 1. Test Workflow (`.github/workflows/test.yml`)

**Trigger**: Every push and pull request

- Runs `flake8` linting on `src/` and `tests/`
- Executes all unit tests with `pytest`
- Uploads coverage report as artifact

### 2. Evaluate Workflow (`.github/workflows/evaluate.yml`)

**Trigger**: After Tests workflow succeeds (on main/master)

- Evaluates model performance on test samples
- Checks if accuracy ≥ 80% threshold
- Uploads evaluation metrics as artifact
- Fails pipeline if below threshold

### 3. Build Workflow (`.github/workflows/build.yml`)

**Trigger**: Push to main/master or version tags

- Builds Docker image
- Publishes to DockerHub registry

**Required GitHub Secrets**:
- `DOCKERHUB_USERNAME` - Your DockerHub username
- `DOCKERHUB_TOKEN` - DockerHub access token
