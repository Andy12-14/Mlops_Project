import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer

from src.data_processing import clean_text, tokenize_text, split_data, process_dataframe


# -----------------------------------------------------------
# TESTS FOR clean_text
# -----------------------------------------------------------

def test_clean_text_lowercase():
    text = "HELLO WORLD!"
    result = clean_text(text)
    assert result == "hello world"


def test_clean_text_remove_urls_emails():
    text = "Check http://example.com and mail me@test.com"
    result = clean_text(text)
    assert "http" not in result and "@" not in result


def test_clean_text_remove_non_letters():
    text = "Hello 123 !!!"
    result = clean_text(text)
    assert result == "hello"


def test_clean_text_collapse_spaces():
    text = "Hello     world   !"
    result = clean_text(text)
    assert result == "hello world"


def test_clean_text_non_string_input():
    result = clean_text(None)
    assert result == ""


# -----------------------------------------------------------
# TESTS FOR tokenize_text
# -----------------------------------------------------------

def test_tokenize_text_output():
    text = "hello world"
    tokens = tokenize_text(text)
    assert "input_ids" in tokens and "attention_mask" in tokens
    assert isinstance(tokens["input_ids"], torch.Tensor)
    assert tokens["input_ids"].ndim == 2  # Should include batch dimension


def test_tokenize_text_expected_ids():
    text = "hello world"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    expected = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    result = tokenize_text(text)["input_ids"]
    assert torch.equal(expected, result)


# -----------------------------------------------------------
# TESTS FOR split_data
# -----------------------------------------------------------

def test_split_data_shapes():
    df = pd.DataFrame({"col": range(10)})
    train_df, val_df = split_data(df, test_size=0.3, random_state=42)
    assert len(train_df) == 7
    assert len(val_df) == 3
    assert not set(train_df.index).intersection(val_df.index)


# -----------------------------------------------------------
# TESTS FOR process_dataframe (mock sentiment)
# -----------------------------------------------------------

@patch("src.data_processing.load_data")
@patch("src.data_processing.pipeline")
def test_process_dataframe(mock_pipeline, mock_load_data, tmp_path):
    # Fake data
    df = pd.DataFrame({
        "content": [
            "HELLO WORLD!!!",
            "This is a test email test@example.com",
            "Visit https://huggingface.co for models",
            ""
        ]
    })
    mock_load_data.return_value = df

    # Fake sentiment pipeline
    fake_sentiment = MagicMock()
    fake_sentiment.side_effect = lambda text: [{"label": "POSITIVE", "score": 0.99}] if text else [{"label": "NEGATIVE", "score": 0.1}]
    mock_pipeline.return_value = fake_sentiment

    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    result = process_dataframe(data_path=csv_path)

    # Basic keys check
    assert "train_df" in result and "val_df" in result and "tokenizer" in result and "stats" in result

    train_df = result["train_df"]

    # Check cleaning
    assert all(train_df["cleaned_content"].apply(lambda x: x == x.lower()))
    assert all("http" not in x for x in train_df["cleaned_content"])

    # Check tokenization columns
    assert "input_ids" in train_df.columns
    assert isinstance(train_df["input_ids"].iloc[0], list)
    assert all(isinstance(i, int) for i in train_df["input_ids"].iloc[0])

    # Check sentiment columns
    assert "sentiment" in train_df.columns
    assert "sentiment_label" in train_df.columns
    assert set(train_df["sentiment_label"].unique()).issubset({0, 1})

    # Check stats
    stats = result["stats"]
    assert stats["total_samples"] == len(df)
    assert stats["train_samples"] + stats["val_samples"] == len(df)
