import os
import numpy as np
import argparse
import torch
from typing import List, Dict, Any
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TrainingArguments,
	Trainer,
	EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from data_processing import process_dataframe


class SentimentDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		# encodings: dict of lists or tensors for input_ids, attention_mask
		self.encodings = {k: (v if isinstance(v, list) else v.tolist()) for k, v in encodings.items()}
		self.labels = list(labels)

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)


class SentimentClassifier:
	def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, cache_dir: str = None):
		self.model_name = model_name
		self.num_labels = num_labels
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Load tokenizer and model
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
		self.model.to(self.device)

	def compute_metrics(self, eval_pred) -> Dict[str, Any]:
		"""
		Compute evaluation metrics including confusion matrix and classification report.
		
		Args:
			eval_pred: Tuple of predictions and labels
			
		Returns:
			Dictionary containing various metrics and plots
		"""
		logits, labels = eval_pred
		preds = np.argmax(logits, axis=1)
		
		# Basic metrics
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
		acc = accuracy_score(labels, preds)
		
		# Confusion Matrix
		cm = confusion_matrix(labels, preds)
		plt.figure(figsize=(8, 6))
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
		disp.plot()
		plt.title('Confusion Matrix')
		
		# Save confusion matrix plot
		os.makedirs('evaluation_plots', exist_ok=True)
		plt.savefig('evaluation_plots/confusion_matrix.png')
		plt.close()
		
		# Detailed classification report
		report = classification_report(labels, preds, target_names=['Negative', 'Positive'], output_dict=True)
		
		# Combine all metrics
		metrics = {
			"accuracy": acc,
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"confusion_matrix": cm.tolist(),
			"classification_report": report
		}
		
		# Print detailed metrics for logging
		print("\nClassification Report:")
		print(classification_report(labels, preds, target_names=['Negative', 'Positive']))
		print("\nConfusion Matrix:")
		print(cm)
		
		return metrics

	def train(self,
			  train_texts: List[str],
			  train_labels: List[int],
			  val_texts: List[str] = None,
			  val_labels: List[int] = None,
			  output_dir: str = "model_outputs",
			  batch_size: int = 16,
			  num_epochs: int = 3,
			  learning_rate: float = 2e-5,
			  warmup_steps: int = 0,
			  weight_decay: float = 0.0,
			  logging_steps: int = 50,
			  eval_steps: int = 200,
			  save_steps: int = 500,
			  early_stopping_patience: int = 2,
			  max_length: int = 128):

		training_args = TrainingArguments(
			output_dir=output_dir,
			num_train_epochs=num_epochs,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			learning_rate=learning_rate,
			warmup_steps=warmup_steps,
			weight_decay=weight_decay,
			logging_steps=logging_steps,
			evaluation_strategy="steps" if val_texts is not None else "no",
			eval_steps=eval_steps,
			save_steps=save_steps,
			load_best_model_at_end=True if val_texts is not None else False,
			metric_for_best_model="f1" if val_texts is not None else None,
			save_total_limit=3,
			fp16=torch.cuda.is_available()
		)

		# Tokenize
		train_enc = self.tokenizer(train_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
		train_dataset = SentimentDataset(train_enc, train_labels)

		val_dataset = None
		if val_texts is not None and val_labels is not None:
			val_enc = self.tokenizer(val_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
			val_dataset = SentimentDataset(val_enc, val_labels)

		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=val_dataset,
			compute_metrics=self.compute_metrics,
			callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if val_dataset is not None else None
		)

		# Train the model
		trainer.train()

		# Save final model using HuggingFace's save_pretrained
		final_dir = os.path.join(output_dir, "final_model")
		os.makedirs(final_dir, exist_ok=True)
		self.model.save_pretrained(final_dir)
		self.tokenizer.save_pretrained(final_dir)
		
		# Save the entire classifier object using joblib
		import joblib
		classifier_path = os.path.join(output_dir, "sentiment_classifier.joblib")
		joblib.dump(self, classifier_path)
		print(f"\nFull classifier saved to: {classifier_path}")
		
		# Save model configuration
		config = {
			"model_name": self.model_name,
			"num_labels": self.num_labels,
			"device": str(self.device),
			"max_length": trainer.args.max_length,
		}
		import json
		config_path = os.path.join(output_dir, "model_config.json")
		with open(config_path, "w") as f:
			json.dump(config, f, indent=4)


def main():
	parser = argparse.ArgumentParser(description="Fine-tune a BERT model on the processed sentiment dataset")
	parser.add_argument("--sample-size", type=int, default=0, help="If >0, use this many training samples (and same proportion for val) for a quick smoke test")
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--output-dir", type=str, default="model_outputs")
	parser.add_argument("--model-name", type=str, default="bert-base-uncased")
	parser.add_argument("--max-length", type=int, default=128)
	args = parser.parse_args()

	print("Loading and processing data (this may download tokenizers/models on first run)...")
	results = process_dataframe()
	train_df = results['train_df']
	val_df = results['val_df']

	# use sentiment_label if present else derive from continuous score
	if 'sentiment_label' in train_df.columns:
		train_labels = train_df['sentiment_label'].astype(int).tolist()
		val_labels = val_df['sentiment_label'].astype(int).tolist()
	elif 'sentiment' in train_df.columns:
		train_labels = (train_df['sentiment'] > 0.5).astype(int).tolist()
		val_labels = (val_df['sentiment'] > 0.5).astype(int).tolist()
	else:
		raise RuntimeError('No sentiment column found in processed data. Run the data pipeline first.')

	train_texts = train_df['cleaned_content'].fillna('').tolist()
	val_texts = val_df['cleaned_content'].fillna('').tolist()

	# Optionally sample for a quick run
	if args.sample_size and args.sample_size > 0:
		n = min(args.sample_size, len(train_texts))
		train_texts = train_texts[:n]
		train_labels = train_labels[:n]
		m = min(max(1, int(n * 0.2)), len(val_texts))
		val_texts = val_texts[:m]
		val_labels = val_labels[:m]

	classifier = SentimentClassifier(model_name=args.model_name, num_labels=2)

	classifier.train(
		train_texts=train_texts,
		train_labels=train_labels,
		val_texts=val_texts,
		val_labels=val_labels,
		output_dir=args.output_dir,
		batch_size=args.batch_size,
		num_epochs=args.epochs,
		max_length=args.max_length
	)


if __name__ == '__main__':
	main()
