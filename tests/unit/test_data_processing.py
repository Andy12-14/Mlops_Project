import pytest
import pandas as pd
import torch
from transformers import AutoTokenizer

# Import functions from your module
from src.data_processing import (
    clean_text,
    tokenize_text,
    split_data,
    process_dataframe
)


# -----------------------------
# TESTS FOR clean_text
# -----------------------------
def test_clean_text_basic_lowercase():
    text = "HELLO WORLD!"
    result = clean_text(text)
    assert result == "hello world", f"Expected lowercase text, got {result}"

def test_clean_text_remove_urls_emails():
    text = "Contact us at info@company.com or visit https://company.com"
    result = clean_text(text)
    assert "http" not in result
    assert "@" not in result
    assert result == "contact us at or visit", f"Unexpected cleaned text: {result}"

def test_clean_text_remove_non_letters():
    text = "Hello 123 !!!"
    result = clean_text(text)
    assert result == "hello", "Numbers and punctuation should be removed"

def test_clean_text_collapse_spaces():
    text = "Hello     world    test"
    result = clean_text(text)
    assert result == "hello world test"

def test_clean_text_non_string_input():
    result = clean_text(None)
    assert result == "", "Non-string input should return empty string"


# -----------------------------
# TESTS FOR tokenize_text
# -----------------------------
def test_tokenize_text_output_format():
    text = "Hello world"
    tokenized = tokenize_text(text)
    assert "input_ids" in tokenized and "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], torch.Tensor)
    assert tokenized["input_ids"].ndim == 2, "Should return batch dimension"

def test_tokenize_text_reproducibility():
    text = "Hello world"
    t1 = tokenize_text(text)
    t2 = tokenize_text(text)
    assert torch.equal(t1["input_ids"], t2["input_ids"]), "Token IDs should be reproducible"


# -----------------------------
# TESTS FOR split_data
# -----------------------------
def test_split_data_shape():
    df = pd.DataFrame({"a": range(10)})
    train_df, val_df = split_data(df, test_size=0.2, random_state=42)
    assert len(train_df) == 8
    assert len(val_df) == 2
    assert not train_df.index.equals(val_df.index), "Train and validation indices should be different"


# -----------------------------
# TESTS FOR process_dataframe
# -----------------------------
@pytest.fixture
def mock_data(tmp_path):
    """Create a temporary CSV dataset for testing."""
    data = pd.DataFrame({
        "content": ["Hello WORLD!", "Second line with EMAIL test@test.com"],
        "replyContent": ["Reply text", "Another reply http://link.com"]
    })
    csv_path = tmp_path / "dataset.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


def test_process_dataframe_end_to_end(mock_data):
    results = process_dataframe(data_path=mock_data)

    # Check structure of the output dict
    for key in ["train_df", "val_df", "tokenizer", "stats"]:
        assert key in results, f"Missing key {key} in results"

    # Check DataFrame contents
    train_df = results["train_df"]
    assert "cleaned_content" in train_df.columns
    assert "input_ids" in train_df.columns
    assert isinstance(train_df.iloc[0]["input_ids"], list)
    assert all(isinstance(i, int) for i in train_df.iloc[0]["input_ids"])

    # Check tokenizer
    tokenizer = results["tokenizer"]
    assert isinstance(tokenizer, AutoTokenizer), "Tokenizer must be HuggingFace AutoTokenizer"

    # Check stats consistency
    stats = results["stats"]
    assert stats["total_samples"] == 2
    assert stats["train_samples"] + stats["val_samples"] == 2
    assert stats["avg_text_length"] > 0
    assert "empty_texts" in stats


def test_process_dataframe_cleaning(mock_data):
    """Ensure cleaning is applied correctly during processing."""
    results = process_dataframe(data_path=mock_data)
    df = pd.concat([results["train_df"], results["val_df"]])
    assert df["cleaned_content"].iloc[0].islower(), "Cleaned text should be lowercase"
    assert "http" not in df["cleaned_replyContent"].iloc[1], "URLs should be removed"
 


