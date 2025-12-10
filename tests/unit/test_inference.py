import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np
from src.inference import SentimentPredictor


@pytest.fixture
def mock_classifier():
    """Create a mock SentimentClassifier with a dummy tokenizer and model."""
    mock_classifier = MagicMock()
    
    # Mock tokenizer behavior - needs .to() method like BatchEncoding
    mock_encoded = MagicMock()
    mock_encoded.__getitem__ = lambda self, key: torch.tensor([[1, 2, 3]]) if key == "input_ids" else torch.tensor([[1, 1, 1]])
    mock_encoded.to = MagicMock(return_value=mock_encoded)
    mock_classifier.tokenizer.return_value = mock_encoded

    # Mock model output (logits)
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[2.0, 0.5]])  # positive > negative
    mock_classifier.model.return_value = mock_outputs
    mock_classifier.model.eval = MagicMock()
    mock_classifier.model.to = MagicMock()
    return mock_classifier


@patch("src.inference.joblib.load")
def test_inference_initialization(mock_joblib_load, mock_classifier):
    """Test that SentimentPredictor correctly loads the classifier."""
    mock_joblib_load.return_value = mock_classifier
    predictor = SentimentPredictor("fake/path/to/model.joblib")

    assert predictor.classifier == mock_classifier
    predictor.classifier.model.eval.assert_called_once()
    predictor.classifier.model.to.assert_called_once()
    assert isinstance(predictor.device, torch.device)


@patch("src.inference.joblib.load")
def test_predict_single_text(mock_joblib_load, mock_classifier):
    """Test prediction output for a single text input."""
    mock_joblib_load.return_value = mock_classifier
    predictor = SentimentPredictor("fake/model/path")

    # Run prediction
    result = predictor.predict("I love AI!")

    assert isinstance(result, list)
    assert len(result) == 1
    output = result[0]

    # Verify output structure
    assert set(output.keys()) == {"text", "sentiment", "confidence", "probabilities"}
    assert output["sentiment"] in ["Positive", "Negative"]
    assert 0.0 <= output["confidence"] <= 1.0
    assert all(0.0 <= p <= 1.0 for p in output["probabilities"].values())


@patch("src.inference.joblib.load")
def test_predict_multiple_texts(mock_joblib_load, mock_classifier):
    """Test prediction for multiple inputs at once."""
    mock_joblib_load.return_value = mock_classifier
    
    # Override model output to return 2 rows for 2 texts
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[2.0, 0.5], [0.3, 1.8]])  # 2 predictions
    mock_classifier.model.return_value = mock_outputs
    
    predictor = SentimentPredictor("fake/model/path")

    texts = ["Good movie", "Bad service"]
    results = predictor.predict(texts)

    assert len(results) == len(texts)
    for r in results:
        assert "sentiment" in r
        assert "confidence" in r
        assert isinstance(r["confidence"], float)
        assert 0.0 <= r["confidence"] <= 1.0
