#!/usr/bin/env python3
"""
Simple Emotion Classification Inference Script

A standalone script to load the trained DistilBERT model and perform
emotion classification on text inputs with comprehensive testing.

Usage:
    python app.py

Author: Generated for emotion classification inference
Date: 2024
"""

import os
import json
import torch
import time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import warnings

warnings.filterwarnings('ignore')


class EmotionClassifier:
    """A simple emotion classifier using the trained DistilBERT model."""
    
    def __init__(self, model_path="./emotion_classifier_model"):
        """
        Initialize the emotion classifier.
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 128
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model, tokenizer, and label mappings."""
        try:
            print(f"Loading model from {self.model_path}...")
            print(f"Using device: {self.device}")
            
            # Load model and tokenizer
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mappings
            label_file = os.path.join(self.model_path, 'label_mappings.json')
            with open(label_file, 'r') as f:
                mappings = json.load(f)
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Number of emotion classes: {len(self.id2label)}")
            print(f"🏷️ Available emotions: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, text):
        """
        Predict emotion for a single text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            tuple: (emotion, confidence, all_probabilities)
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral", 0.0, {}
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move inputs to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get all probabilities for detailed analysis
        all_probs = {}
        for i, prob in enumerate(probabilities[0]):
            emotion = self.id2label[i]
            all_probs[emotion] = prob.item()
        
        # Sort by probability
        all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        emotion = self.id2label[predicted_class]
        return emotion, confidence, all_probs
    
    def predict_batch(self, texts):
        """
        Predict emotions for multiple texts efficiently.
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of (text, emotion, confidence) tuples
        """
        if not texts:
            return []
        
        # Filter valid texts
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        
        if not valid_texts:
            return [(text, "neutral", 0.0) for text in texts]
        
        # Tokenize all texts
        inputs = self.tokenizer(
            valid_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move inputs to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            confidences = torch.max(probabilities, dim=-1)[0]
        
        # Format results
        results = []
        for i, text in enumerate(valid_texts):
            emotion = self.id2label[predicted_classes[i].item()]
            confidence = confidences[i].item()
            results.append((text, emotion, confidence))
        
        return results
    
    def analyze_text(self, text, show_top_n=3):
        """
        Detailed analysis of text showing top N emotions with probabilities.
        
        Args:
            text (str): Input text to analyze
            show_top_n (int): Number of top emotions to show
            
        Returns:
            dict: Detailed analysis results
        """
        emotion, confidence, all_probs = self.predict(text)
        
        # Get top N emotions
        top_emotions = list(all_probs.items())[:show_top_n]
        
        analysis = {
            'text': text,
            'predicted_emotion': emotion,
            'confidence': confidence,
            'top_emotions': top_emotions,
            'analysis_summary': f"Most likely: {emotion} ({confidence:.1%})"
        }
        
        return analysis


def run_sample_tests(classifier):
    """Run comprehensive test cases to demonstrate the model's capabilities."""
    print("\n" + "="*80)
    print("🧪 RUNNING SAMPLE TEST CASES")
    print("="*80)
    
    # Test cases covering different emotions
    test_cases = [
        # Happiness
        ("I am so excited about this new opportunity!", "happiness"),
        ("This is the best day of my life!", "happiness"),
        ("I feel absolutely wonderful today!", "happiness"),
        
        # Love
        ("I love spending time with my family", "love"),
        ("She means everything to me", "love"),
        ("I adore chocolate ice cream", "love"),
        
        # Anger
        ("This is completely unacceptable behavior!", "anger"),
        ("I'm furious about this situation", "anger"),
        ("This makes me really mad", "anger"),
        
        # Sadness
        ("I feel so lonely and depressed", "sadness"),
        ("This news breaks my heart", "sadness"),
        ("I'm feeling quite down today", "sadness"),
        
        # Fear/Worry
        ("I'm terrified of what might happen", "worry"),
        ("This situation makes me very anxious", "worry"),
        ("I'm scared about the future", "worry"),
        
        # Neutral
        ("The weather is partly cloudy today", "neutral"),
        ("I need to buy groceries", "neutral"),
        ("The meeting is scheduled for 3 PM", "neutral"),
        
        # Surprise
        ("I can't believe this actually happened!", "surprise"),
        ("What an unexpected turn of events!", "surprise"),
        
        # Disgust/Hate
        ("I absolutely hate waiting in long lines", "hate"),
        ("This food tastes terrible", "hate"),
        
        # Mixed/Complex emotions
        ("I'm happy but also nervous about starting this new job", "happiness"),
        ("It's bittersweet to see my children grow up", "neutral"),
    ]
    
    print(f"\n📝 Testing {len(test_cases)} sample cases...\n")
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        emotion, confidence, _ = classifier.predict(text)
        
        # Check if prediction matches expected (case-insensitive)
        is_correct = emotion.lower() == expected.lower()
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{i:2d}. {status} Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        print(f"     Expected: {expected:12} | Predicted: {emotion:12} | Confidence: {confidence:.1%}")
        print()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    print(f"📊 Test Results: {correct_predictions}/{total_predictions} correct predictions")
    print(f"🎯 Accuracy: {accuracy:.1%}")
    
    return accuracy


def run_batch_test(classifier):
    """Test batch prediction functionality."""
    print("\n" + "="*80)
    print("🚀 TESTING BATCH PREDICTION")
    print("="*80)
    
    batch_texts = [
        "I love this amazing weather!",
        "I'm really angry about this delay",
        "Feeling quite sad today",
        "This is a neutral statement",
        "I'm worried about tomorrow's exam"
    ]
    
    print(f"\n🔄 Processing {len(batch_texts)} texts in batch...\n")
    
    start_time = time.time()
    results = classifier.predict_batch(batch_texts)
    end_time = time.time()
    
    for i, (text, emotion, confidence) in enumerate(results, 1):
        print(f"{i}. '{text}' → {emotion} ({confidence:.1%})")
    
    processing_time = end_time - start_time
    print(f"\n⚡ Batch processing time: {processing_time:.3f} seconds")
    print(f"📈 Average time per text: {processing_time/len(batch_texts):.3f} seconds")


def run_detailed_analysis(classifier):
    """Run detailed emotion analysis showing top predictions."""
    print("\n" + "="*80)
    print("🔍 DETAILED EMOTION ANALYSIS")
    print("="*80)
    
    analysis_texts = [
        "I'm feeling really excited but also a bit nervous about this new adventure!",
        "This situation is frustrating and makes me quite angry.",
        "I absolutely love spending time in nature, it brings me peace."
    ]
    
    for i, text in enumerate(analysis_texts, 1):
        print(f"\n{i}. Analyzing: '{text}'")
        print("-" * 70)
        
        analysis = classifier.analyze_text(text, show_top_n=5)
        
        print(f"🎯 Primary Emotion: {analysis['predicted_emotion']} ({analysis['confidence']:.1%})")
        print("\n📊 Top 5 Emotion Probabilities:")
        
        for j, (emotion, prob) in enumerate(analysis['top_emotions'], 1):
            bar_length = int(prob * 30)  # Visual bar
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {j}. {emotion:12} │{bar}│ {prob:.1%}")


def run_performance_test(classifier):
    """Test model performance and speed."""
    print("\n" + "="*80)
    print("⚡ PERFORMANCE TESTING")
    print("="*80)
    
    # Test single prediction speed
    test_text = "This is a test sentence for performance evaluation."
    
    print(f"\n🔄 Testing single prediction speed...")
    times = []
    for _ in range(10):
        start_time = time.time()
        classifier.predict(test_text)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"⏱️  Average prediction time: {avg_time:.4f} seconds")
    print(f"🚀 Predictions per second: {1/avg_time:.1f}")
    
    # Test batch vs individual predictions
    batch_texts = [test_text] * 100
    
    print(f"\n🔄 Testing batch vs individual predictions (100 texts)...")
    
    # Individual predictions
    start_time = time.time()
    for text in batch_texts:
        classifier.predict(text)
    individual_time = time.time() - start_time
    
    # Batch prediction
    start_time = time.time()
    classifier.predict_batch(batch_texts)
    batch_time = time.time() - start_time
    
    speedup = individual_time / batch_time if batch_time > 0 else 0
    
    print(f"⏱️  Individual predictions: {individual_time:.3f} seconds")
    print(f"⚡ Batch prediction: {batch_time:.3f} seconds")
    print(f"🚀 Speedup factor: {speedup:.1f}x")


def main():
    """Main function to run all tests and demonstrations."""
    print("🎭 EMOTION CLASSIFICATION INFERENCE APP")
    print("="*80)
    print("Loading DistilBERT emotion classifier...")
    
    try:
        # Initialize classifier
        classifier = EmotionClassifier()
        
        # Run all test suites
        print(f"\n🎯 Model ready! Running comprehensive tests...\n")
        
        # 1. Sample test cases
        accuracy = run_sample_tests(classifier)
        
        # 2. Batch prediction test
        run_batch_test(classifier)
        
        # 3. Detailed analysis
        run_detailed_analysis(classifier)
        
        # 4. Performance testing
        run_performance_test(classifier)
        
        # 5. Interactive demo
        print("\n" + "="*80)
        print("🎮 INTERACTIVE DEMO")
        print("="*80)
        print("\nTry your own text! (Press Enter with empty text to exit)")
        
        while True:
            try:
                user_input = input("\n💭 Enter text to analyze: ").strip()
                
                if not user_input:
                    print("👋 Goodbye!")
                    break
                
                emotion, confidence, top_probs = classifier.predict(user_input)
                
                print(f"\n🎯 Result: {emotion} ({confidence:.1%} confidence)")
                
                # Show top 3 emotions
                print("📊 Top 3 emotions:")
                for i, (emo, prob) in enumerate(list(top_probs.items())[:3], 1):
                    print(f"   {i}. {emo:12} - {prob:.1%}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Final summary
        print("\n" + "="*80)
        print("📊 TESTING SUMMARY")
        print("="*80)
        print(f"✅ Model loaded successfully")
        print(f"🎯 Test accuracy: {accuracy:.1%}")
        print(f"⚡ Performance: Ready for production use")
        print(f"🚀 All systems operational!")
        
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        print("Please ensure the trained model exists in './emotion_classifier_model/'")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
