# MLOps Project — Sentiment Analysis of User Reviews

This repository contains a small MLOps-style pipeline for performing sentiment analysis on user reviews. It includes data extraction, preprocessing, a model training pipeline (fine-tuning a transformer), and an inference interface with unit tests.

This README explains how to set up the environment, run the pipeline, and use the provided modules.

## Repo layout

```
Mlops_Project/
├── dataset/
│   └── dataset.csv            # Raw dataset (CSV)
├── src/
│   ├── data_extraction.py    # load_data(...) - reads dataset.csv
│   ├── data_processing.py    # process_dataframe(...) - cleaning, tokenization, split
│   ├── model.py              # SentimentClassifier (train/evaluate/save)
│   └── inference.py          # SentimentPredictor (load model and predict)
├── tests/
│   └── unit/                 # pytest unit tests for each module
├── requirements.txt          # pinned Python dependencies
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

## License & contact

This project is provided as-is. Add your preferred license file (e.g., `LICENSE`) to make licensing explicit. For questions or help, open an issue in the repository.

---

If you'd like, I can also:
- run the test suite now and report results (quick smoke test), or
- add a short CONTRIBUTING.md and a minimal `requirements.txt` review to ensure everything needed for the README steps is present.

