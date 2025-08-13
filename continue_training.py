#!/usr/bin/env python3
"""
Continue Training Script for DistilBERT Emotion Classification

This script continues training from an existing model checkpoint to potentially
improve performance further.

Usage:
    python continue_training.py
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import json
import warnings

warnings.filterwarnings('ignore')


class MemoryEfficientEmotionDataset(Dataset):
    """Memory-efficient Dataset class that tokenizes on-the-fly."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)


def load_existing_model(model_path):
    """Load the existing trained model and its configurations."""
    print(f"Loading existing model from {model_path}...")
    
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    
    # Load label mappings
    with open(f'{model_path}/label_mappings.json', 'r') as f:
        mappings = json.load(f)
        label_map = mappings['label_map']
        id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    print(f"✅ Loaded model with {len(label_map)} emotion classes")
    return model, tokenizer, label_map, id2label


def prepare_data(csv_path, label_map, sample_size=None, test_size=0.2, val_size=0.1):
    """Prepare training data with the same preprocessing as original training."""
    print(f"Loading dataset from {csv_path}...")
    
    # Load data with optional sampling
    if sample_size:
        print(f"Sampling {sample_size} rows from the dataset...")
        total_rows = sum(1 for line in open(csv_path)) - 1
        if total_rows > sample_size:
            skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                              total_rows - sample_size, 
                                              replace=False))
            data = pd.read_csv(csv_path, skiprows=skip_rows)
        else:
            data = pd.read_csv(csv_path)
    else:
        data = pd.read_csv(csv_path)
    
    # Preprocess data
    data = data.drop_duplicates()
    data['text'] = data['text'].astype(str).str.lower().str.strip()
    data = data.dropna(subset=['text', 'Emotion'])
    
    # Map emotions to labels
    data['label'] = data['Emotion'].map(label_map)
    
    # Remove any unmapped emotions (if new emotions exist)
    data = data.dropna(subset=['label'])
    data['label'] = data['label'].astype(int)
    
    print(f"Dataset shape after preprocessing: {data.shape}")
    
    # Create splits
    train_val, test_data = train_test_split(
        data, test_size=test_size, stratify=data['label'], random_state=42
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    train_data, val_data = train_test_split(
        train_val, test_size=val_size_adjusted, stratify=train_val['label'], random_state=42
    )
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    return train_data, val_data, test_data


def continue_training(model, tokenizer, train_data, val_data, label_map, id2label,
                     epochs=2, batch_size=8, learning_rate=1e-5, output_dir="./continued_results"):
    """Continue training the existing model."""
    print("Setting up continued training...")
    
    # Create datasets
    train_dataset = MemoryEfficientEmotionDataset(
        train_data['text'].reset_index(drop=True),
        train_data['label'].reset_index(drop=True),
        tokenizer, 128
    )
    
    val_dataset = MemoryEfficientEmotionDataset(
        val_data['text'].reset_index(drop=True),
        val_data['label'].reset_index(drop=True),
        tokenizer, 128
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Using device: {device}")
    
    # Training arguments with lower learning rate for fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        warmup_steps=100,  # Less warmup for continued training
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=None,
        seed=42,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,  # Lower learning rate for fine-tuning
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    
    # Metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print(f"🚀 Starting continued training for {epochs} epochs...")
    print(f"📚 Learning rate: {learning_rate}")
    print(f"🔄 Batch size: {batch_size}")
    
    # Start training
    trainer.train()
    
    print("✅ Continued training completed!")
    return trainer


def evaluate_continued_model(trainer, test_data, tokenizer, id2label, output_dir):
    """Evaluate the continued training results."""
    print("Evaluating continued training results...")
    
    # Create test dataset
    test_dataset = MemoryEfficientEmotionDataset(
        test_data['text'].reset_index(drop=True),
        test_data['label'].reset_index(drop=True),
        tokenizer, 128
    )
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_data['label'].tolist()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 New Test Accuracy: {accuracy:.4f}")
    
    # Save results
    with open(f'{output_dir}/continued_training_results.txt', 'w') as f:
        f.write(f"Continued Training Results\n")
        f.write(f"========================\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Continue training DistilBERT emotion classifier')
    parser.add_argument('--model_path', type=str, default='./emotion_classifier_model',
                       help='Path to existing trained model')
    parser.add_argument('--csv_path', type=str, default='EmotionDetection.csv',
                       help='Path to the CSV dataset')
    parser.add_argument('--output_dir', type=str, default='./continued_results',
                       help='Output directory for continued training')
    parser.add_argument('--new_model_dir', type=str, default='./emotion_classifier_model_v2',
                       help='Directory to save the improved model')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Additional epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size (None for full dataset)')
    
    args = parser.parse_args()
    
    try:
        print("🔄 CONTINUING DISTILBERT TRAINING")
        print("="*60)
        
        # Load existing model
        model, tokenizer, label_map, id2label = load_existing_model(args.model_path)
        
        # Prepare data
        train_data, val_data, test_data = prepare_data(
            args.csv_path, label_map, args.sample_size
        )
        
        # Continue training
        trainer = continue_training(
            model, tokenizer, train_data, val_data, label_map, id2label,
            args.epochs, args.batch_size, args.learning_rate, args.output_dir
        )
        
        # Evaluate results
        new_accuracy = evaluate_continued_model(
            trainer, test_data, tokenizer, id2label, args.output_dir
        )
        
        # Save improved model
        print(f"💾 Saving improved model to {args.new_model_dir}...")
        trainer.save_model(args.new_model_dir)
        tokenizer.save_pretrained(args.new_model_dir)
        
        # Save label mappings
        with open(f'{args.new_model_dir}/label_mappings.json', 'w') as f:
            json.dump({
                'label_map': label_map,
                'id2label': id2label
            }, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 CONTINUED TRAINING SUMMARY")
        print("="*60)
        print(f"✅ Additional training completed")
        print(f"🎯 New accuracy: {new_accuracy:.4f}")
        print(f"💾 Improved model saved to: {args.new_model_dir}")
        print(f"📁 Training logs saved to: {args.output_dir}")
        
        if new_accuracy > 0.995:  # If accuracy improved from 99.51%
            print(f"🎉 Congratulations! Model performance improved!")
        else:
            print(f"ℹ️  Model may have reached optimal performance")
        
    except Exception as e:
        print(f"❌ Error during continued training: {e}")
        raise


if __name__ == "__main__":
    main()
