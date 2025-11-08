import joblib
import argparse
from typing import List, Union
import torch
from data_processing import clean_text
import numpy as np


class SentimentPredictor:
    def __init__(self, model_path: str = "model_outputs/sentiment_classifier.joblib"):
        """
        Initialize the sentiment predictor with a trained model.
        
        Args:
            model_path: Path to the saved classifier model
        """
        print(f"Loading model from {model_path}...")
        self.classifier = joblib.load(model_path)
        
        # Ensure model is in evaluation mode
        self.classifier.model.eval()
        
        # Move model to available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.model.to(self.device)
        print(f"Model loaded successfully and running on {self.device}")

    def predict(self, texts: Union[str, List[str]], clean_texts: bool = True) -> List[dict]:
        """
        Predict sentiment for one or more texts.
        
        Args:
            texts: Single text string or list of texts
            clean_texts: Whether to clean the texts before prediction
            
        Returns:
            List of dictionaries containing predictions for each text
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Clean texts if requested
        if clean_texts:
            texts = [clean_text(text) for text in texts]
            
        # Tokenize
        encoded = self.classifier.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.classifier.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(outputs.logits, dim=1)
            
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()
        preds = predictions.cpu().numpy()
        
        # Prepare results
        results = []
        for text, prob, pred in zip(texts, probs, preds):
            results.append({
                "text": text,
                "sentiment": "Positive" if pred == 1 else "Negative",
                "confidence": float(prob[pred]),  # Probability of predicted class
                "probabilities": {
                    "negative": float(prob[0]),
                    "positive": float(prob[1])
                }
            })
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for text input")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="Path to file containing texts (one per line)")
    parser.add_argument("--model-path", type=str, default="model_outputs/sentiment_classifier.joblib",
                      help="Path to the trained model")
    parser.add_argument("--output", type=str, help="Path to save predictions (JSON format)")
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("Please provide either --text or --file")
        
    # Initialize predictor
    predictor = SentimentPredictor(args.model_path)
    
    # Get texts to predict
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]
        
    # Get predictions
    results = predictor.predict(texts)
    
    # Save or print results
    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nPredictions saved to {args.output}")
    else:
        print("\nPredictions:")
        for result in results:
            print(f"\nText: {result['text'][:100]}...")
            print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")
            print(f"Probabilities: Negative: {result['probabilities']['negative']:.2%}, "
                  f"Positive: {result['probabilities']['positive']:.2%}")


if __name__ == "__main__":
    main()
