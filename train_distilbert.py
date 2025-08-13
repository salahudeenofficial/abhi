#!/usr/bin/env python3
"""
DistilBERT Emotion Classification Training Script

This script fine-tunes a DistilBERT model for emotion classification
using a CSV dataset with text and emotion columns.

Author: Generated for emotion classification task
Date: 2024
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')


class EmotionDataset(Dataset):
    """Custom Dataset class for emotion classification."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_and_preprocess_data(csv_path, text_column='text', label_column='Emotion'):
    """Load and preprocess the dataset."""
    print(f"Loading dataset from {csv_path}...")
    
    # Load data
    data = pd.read_csv(csv_path)
    print(f"Original dataset shape: {data.shape}")
    
    # Check if columns exist
    if text_column not in data.columns or label_column not in data.columns:
        available_columns = list(data.columns)
        raise ValueError(f"Columns '{text_column}' or '{label_column}' not found. Available columns: {available_columns}")
    
    # Basic preprocessing
    data = data.drop_duplicates()
    data[text_column] = data[text_column].astype(str).str.lower().str.strip()
    data = data.dropna(subset=[text_column, label_column])
    
    print(f"Dataset shape after preprocessing: {data.shape}")
    
    # Create label mappings
    unique_emotions = sorted(data[label_column].unique())
    label_map = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    id2label = {idx: emotion for emotion, idx in label_map.items()}
    data['label'] = data[label_column].map(label_map)
    
    print(f"Number of classes: {len(label_map)}")
    print(f"Emotion distribution:")
    print(data[label_column].value_counts())
    
    return data, label_map, id2label


def create_data_splits(data, test_size=0.2, val_size=0.1, random_state=42):
    """Create train, validation, and test splits."""
    # First split: train + val vs test
    train_val, test_data = train_test_split(
        data, 
        test_size=test_size, 
        stratify=data['label'], 
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_data, val_data = train_test_split(
        train_val, 
        test_size=val_size_adjusted, 
        stratify=train_val['label'], 
        random_state=random_state
    )
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    return train_data, val_data, test_data


def tokenize_data(train_data, val_data, test_data, model_name='distilbert-base-uncased', max_length=128):
    """Tokenize the datasets."""
    print("Initializing tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    def tokenize_function(texts):
        return tokenizer(
            list(texts), 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
    
    print("Tokenizing datasets...")
    train_encodings = tokenize_function(train_data['text'])
    val_encodings = tokenize_function(val_data['text'])
    test_encodings = tokenize_function(test_data['text'])
    
    # Create dataset objects
    train_dataset = EmotionDataset(train_encodings, train_data['label'].tolist())
    val_dataset = EmotionDataset(val_encodings, val_data['label'].tolist())
    test_dataset = EmotionDataset(test_encodings, test_data['label'].tolist())
    
    return train_dataset, val_dataset, test_dataset, tokenizer


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}


def train_model(train_dataset, val_dataset, num_labels, id2label, label_map, 
                model_name='distilbert-base-uncased', output_dir='./results',
                num_epochs=3, batch_size=16, learning_rate=2e-5):
    """Train the DistilBERT model."""
    print("Initializing model...")
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map
    )
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=None,
        seed=42,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    return trainer, model


def evaluate_model(trainer, test_dataset, test_data, id2label, save_dir='./results'):
    """Evaluate the trained model on test set."""
    print("Evaluating model on test set...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_data['label'].tolist()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(f'{save_dir}/classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, report


def save_model_and_artifacts(trainer, tokenizer, label_map, id2label, save_dir='./emotion_classifier_model'):
    """Save the trained model and associated artifacts."""
    print(f"Saving model and artifacts to {save_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model and tokenizer
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save label mappings
    with open(f'{save_dir}/label_mappings.json', 'w') as f:
        json.dump({
            'label_map': label_map,
            'id2label': id2label
        }, f, indent=2)
    
    print(f"Model saved successfully to {save_dir}")


def predict_emotion(text, model_path, device='cpu'):
    """Load model and predict emotion for given text."""
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    
    # Load label mappings
    with open(f'{model_path}/label_mappings.json', 'r') as f:
        mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    emotion = id2label[predicted_class]
    return emotion, confidence


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT for emotion classification')
    parser.add_argument('--csv_path', type=str, default='EmotionDetection.csv',
                       help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of the text column in CSV')
    parser.add_argument('--label_column', type=str, default='Emotion',
                       help='Name of the label column in CSV')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='Pre-trained model name')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for training results')
    parser.add_argument('--model_save_dir', type=str, default='./emotion_classifier_model',
                       help='Directory to save the final model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (proportion)')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size (proportion)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    try:
        # Load and preprocess data
        data, label_map, id2label = load_and_preprocess_data(
            args.csv_path, args.text_column, args.label_column
        )
        
        # Create data splits
        train_data, val_data, test_data = create_data_splits(
            data, args.test_size, args.val_size, args.seed
        )
        
        # Tokenize data
        train_dataset, val_dataset, test_dataset, tokenizer = tokenize_data(
            train_data, val_data, test_data, args.model_name, args.max_length
        )
        
        # Train model
        trainer, model = train_model(
            train_dataset, val_dataset, len(label_map), id2label, label_map,
            args.model_name, args.output_dir, args.epochs, args.batch_size, args.learning_rate
        )
        
        # Evaluate model
        accuracy, report = evaluate_model(trainer, test_dataset, test_data, id2label, args.output_dir)
        
        # Save model and artifacts
        save_model_and_artifacts(trainer, tokenizer, label_map, id2label, args.model_save_dir)
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {accuracy:.4f}")
        print(f"Model saved to: {args.model_save_dir}")
        print(f"Training results saved to: {args.output_dir}")
        
        # Test inference with sample texts
        print("\nTesting inference on sample texts:")
        test_texts = [
            "I am so happy today!",
            "This makes me really angry.",
            "I feel sad and lonely.",
            "I'm scared of what might happen.",
            "This is amazing, I love it!"
        ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for text in test_texts:
            try:
                emotion, confidence = predict_emotion(text, args.model_save_dir, device)
                print(f"Text: '{text}' -> Emotion: {emotion} (Confidence: {confidence:.3f})")
            except Exception as e:
                print(f"Error predicting for '{text}': {e}")
                
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
