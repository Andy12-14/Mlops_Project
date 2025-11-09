from data_extraction import load_data
import pandas as pd
import os
import re
import numpy as np
from transformers import AutoTokenizer, pipeline
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Enable progress bar for pandas operations
tqdm.pandas()

# Build dataset path relative to this file so the script works when run
# from the `src/` directory or from the project root
here = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(here), 'dataset', 'dataset.csv')


def clean_text(text):
    """
    Cleans a given text string by removing unnecessary characters,
    lowercasing, and normalization.

    Args:
        text: The input text string.

    Returns:
        The cleaned text string.
    """
    if isinstance(text, str):
        # Lowercase conversion
        text = text.lower()
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+|mailto:\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        # Remove non-letter characters (keep spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ""


def tokenize_text(text, tokenizer_name="bert-base-uncased", max_length=128):
    """
    Tokenizes the input text using a Hugging Face AutoTokenizer.

    Args:
        text: The input text string.
        tokenizer_name: The name of the pre-trained tokenizer to use.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        The tokenized output with input_ids and attention_mask.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_output = tokenizer(
        text, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return tokenized_output


def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the input DataFrame into training and validation sets.

    Args:
        df: The input pandas DataFrame.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
        A tuple containing the training and validation DataFrames.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, val_df


def process_dataframe(
    data_path=None,
    text_column='content',
    tokenizer_name="bert-base-uncased",
    max_length=128,
    test_size=0.2,
    random_state=42
):
    """
    Main processing function that loads data, cleans text, tokenizes, and splits into train/val.
    
    Args:
        data_path: Path to the dataset CSV. If None, uses default path.
        text_column: Name of the main text column to process
        tokenizer_name: Name of the HuggingFace tokenizer to use.
        max_length: Maximum sequence length for tokenization.
        test_size: Proportion of data to use for validation.
        random_state: Random seed for reproducibility.
    
    Returns:
        Dict containing:
        - train_df: Training DataFrame
        - val_df: Validation DataFrame
        - tokenizer: The loaded tokenizer
        - stats: Dict with processing statistics
    """
    # Use default path if none provided
    if data_path is None:
        data_path = dataset_path
    
    # Load and select columns
    df = load_data(data_path)
    text_cols = [col for col in [text_column] if col in df.columns]
    df = df[text_cols].copy()
    
    # Clean text columns
    df[f'cleaned_{text_column}'] = df[text_column].apply(clean_text)
   
    
    # Get tokenizer once for batch processing
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Batch tokenize main content
    texts = df[f'cleaned_{text_column}'].fillna('').tolist()
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Add tokenized outputs to DataFrame
    df['input_ids'] = tokenized['input_ids'].tolist()
    df['attention_mask'] = tokenized['attention_mask'].tolist()
    
    # Initialize sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    print("Processing sentiments...")
    # Process sentiments in batches for efficiency
    def get_sentiment_score(text):
        if not text:
            return 0.5  # neutral score for empty text
        try:
            result = sentiment_pipeline(text)[0]
            # Convert POSITIVE/NEGATIVE to numeric score
            # POSITIVE -> 1.0, NEGATIVE -> 0.0
            return 1.0 if result['label'] == 'POSITIVE' else 0.0
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return 0.5  # neutral score for errors
    
    # Apply sentiment analysis
    df['sentiment'] = df[f'cleaned_{text_column}'].progress_apply(get_sentiment_score)
    
    # Add sentiment labels (can be used for binary classification)
    df['sentiment_label'] = (df['sentiment'] > 0.5).astype(int)


    # Split into train/val
    train_df, val_df = split_data(df, test_size=test_size, random_state=random_state)
    df[f'cleaned_{text_column}'][df[f'cleaned_{text_column}'] != '']
    # Collect statistics
    stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'avg_text_length': df[f'cleaned_{text_column}'].str.len().mean(),
        'max_text_length': df[f'cleaned_{text_column}'].str.len().max(),
        'empty_texts': (df[f'cleaned_{text_column}'] == '').sum()
    }
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'tokenizer': tokenizer,
        'stats': stats
    }


if __name__ == '__main__':
    # Process the dataset and get results
    results = process_dataframe()
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {results['stats']['total_samples']}")
    print(f"Training samples: {results['stats']['train_samples']}")
    print(f"Validation samples: {results['stats']['val_samples']}")
    print(f"\nText Statistics:")
    print(f"Average text length: {results['stats']['avg_text_length']:.1f} chars")
    print(f"Maximum text length: {results['stats']['max_text_length']} chars")
    print(f"Empty texts: {results['stats']['empty_texts']}")
    
    # Print sentiment distribution
    train_sentiments = results['train_df']['sentiment_label'].value_counts()
    print("\nSentiment Distribution in Training Set:")
    print(f"Positive samples: {train_sentiments.get(1, 0)}")
    print(f"Negative samples: {train_sentiments.get(0, 0)}")
    
    # Show a few examples with sentiments
    print("\nFirst few training examples with sentiments:")
    print(results['train_df'][['cleaned_content', 'sentiment', 'sentiment_label']].head())