import torch
import pytest
from transformers import AutoTokenizer
from src.model import SentimentClassifier

@pytest.fixture(scope="module")
def classifier():
    """Instantiate the SentimentClassifier (no training)."""
    return SentimentClassifier(model_name="bert-base-uncased", num_labels=2)

@pytest.fixture(scope="module")
def tokenizer(classifier):
    """Return the classifier's tokenizer for text preprocessing."""
    return classifier.tokenizer

def test_model_instantiation(classifier):
    """Ensure the model initializes correctly and moves to the right device."""
    assert classifier.model is not None, "Model should be initialized"
    assert classifier.tokenizer is not None, "Tokenizer should be initialized"
    assert isinstance(classifier.num_labels, int)
    assert next(classifier.model.parameters()).device == classifier.device

def test_tokenizer_output(tokenizer):
    """Check tokenizer produces valid tensors for dummy text."""
    text = ["I love AI", "I hate bugs"]
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    assert "input_ids" in enc
    assert "attention_mask" in enc
    assert enc["input_ids"].shape[0] == len(text)

def test_model_forward_output_shape(classifier, tokenizer):
    """Run a dummy batch through the model and check logits shape."""
    texts = ["Great movie!", "Terrible food."]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(classifier.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = classifier.model(**inputs)

    assert hasattr(outputs, "logits"), "Model output should contain logits"
    logits = outputs.logits

    batch_size = len(texts)
    num_labels = classifier.num_labels
    assert logits.shape == (batch_size, num_labels), (
        f"Expected logits shape {(batch_size, num_labels)}, but got {logits.shape}"
    )

def test_model_compute_metrics_basic(classifier):
    """Verify compute_metrics returns expected keys and correct shapes."""
    # Fake predictions and labels
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.8]]).numpy()
    labels = torch.tensor([0, 1]).numpy()
    metrics = classifier.compute_metrics((logits, labels))

    expected_keys = {"accuracy", "precision", "recall", "f1", "confusion_matrix", "classification_report"}
    assert expected_keys.issubset(metrics.keys()), "Missing expected metric keys"
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["confusion_matrix"], list)
    assert len(metrics["confusion_matrix"]) == 2.0
