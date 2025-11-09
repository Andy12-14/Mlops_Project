# MLOps Project: Sentiment Analysis of User Reviews

This project implements a machine learning pipeline for sentiment analysis of user reviews. The goal is to classify reviews as either positive or negative. The project is structured to follow MLOps best practices, including data processing, model training, and inference.

This project was a collaborative effort by a team of four. Each member worked on their own branch, and all code was peer-reviewed before being merged into the main branch.

## Project Structure

```
.
├── dataset
│   └── dataset.csv
├── src
│   ├── data_extraction.py
│   ├── data_processing.py
│   ├── model.py
│   └── inference.py
├── tests
│   └── unit
│       └── test_data_extraction.py
├── .gitignore
├── README.md
└── requirements.txt
```

- **`dataset/dataset.csv`**: The raw data containing user reviews.
- **`src/data_extraction.py`**: A script to load data from the CSV file.
- **`src/data_processing.py`**: This script handles text cleaning, tokenization, and splitting the data into training and validation sets. It also uses a pre-trained model to generate initial sentiment labels.
- **`src/model.py`**: This script defines the sentiment classification model, a training pipeline, and evaluation metrics. It uses the Hugging Face `transformers` library to fine-tune a pre-trained BERT model.
- **`src/inference.py`**: This script provides a class to load the trained model and perform sentiment prediction on new text.
- **`tests/`**: Contains unit tests for the project.
- **`requirements.txt`**: A list of all the Python packages required to run the project.

## How It Works

The project follows a standard machine learning pipeline:

1.  **Data Extraction**: The `load_data` function in `data_extraction.py` reads the user reviews from `dataset.csv`.
2.  **Data Processing**: The `process_dataframe` function in `data_processing.py` takes the raw data and performs several preprocessing steps:
    - **Text Cleaning**: Removes URLs, email addresses, and other noise from the review text.
    - **Tokenization**: Uses a pre-trained tokenizer from the `transformers` library to convert the text into a format suitable for the model.
    - **Sentiment Labeling**: Uses a pre-trained sentiment analysis model to assign an initial sentiment score and label (positive/negative) to each review.
    - **Data Splitting**: Splits the processed data into training and validation sets.
3.  **Model Training**: The `model.py` script fine-tunes a `bert-base-uncased` model for sequence classification.
    - The `SentimentClassifier` class manages the model, tokenizer, and training process.
    - The `train` method uses the `Trainer` API from the `transformers` library to fine-tune the model on the training data.
    - During training, the model's performance is evaluated on the validation set, and metrics such as accuracy, precision, recall, and F1-score are calculated.
4.  **Inference**: The `inference.py` script is used to make predictions on new, unseen text.
    - The `SentimentPredictor` class loads the fine-tuned model and provides a `predict` method that takes a string or a list of strings and returns the sentiment prediction(s).

## How to Use

### 1. Installation

First, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd MLOps_project
pip install -r requirements.txt
```

### 2. Data Processing

To process the data, run the `data_processing.py` script. This will clean the data, generate sentiment labels, and prepare it for training.

```bash
python src/data_processing.py
```

### 3. Model Training

To train the sentiment analysis model, run the `model.py` script. This will fine-tune the BERT model on the processed data and save the trained model to the `model_outputs` directory.

```bash
python src/model.py
```

You can customize the training process with command-line arguments:

- `--epochs`: Number of training epochs.
- `--batch-size`: Training batch size.
- `--output-dir`: Directory to save the trained model.
- `--model-name`: The name of the pre-trained model to use.

### 4. Inference

To make predictions on new text, use the `inference.py` script. You can provide text directly via the command line or pass a file with a list of texts.

**Predicting a single text:**

```bash
python src/inference.py --text "This is a great app!"
```

**Predicting from a file:**

Create a file (e.g., `texts.txt`) with one text per line:

```
This app is amazing!
I'm not happy with the latest update.
```

Then run the inference script:

```bash
python src/inference.py --file texts.txt
```

The script will output the sentiment ("Positive" or "Negative"), confidence score, and the probabilities for each class.
