"""
SignBridge — Phase 2, Step 1
Model Training: ASL Alphabet Classifier

This script trains a Dense neural network on the extracted hand landmarks
to classify ASL alphabet letters (A-Z + del/nothing/space).

Input: 
    - data/alphabet/landmarks.npy (63 features per sample)
    - data/alphabet/labels.npy (string labels)

Output:
    - models/alphabet_model.h5 (trained Keras model)
    - models/label_encoder.pkl (fitted LabelEncoder for inference)
"""

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configuration
# ==========================================
DATA_DIR = "data/alphabet"
MODEL_DIR = "models"
LANDMARKS_FILE = os.path.join(DATA_DIR, "landmarks.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.npy")
MODEL_FILE = os.path.join(MODEL_DIR, "alphabet_model.h5")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Training hyperparameters
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load landmarks and labels, preprocess for training."""
    print("="*60)
    print("  Loading and Preprocessing Data")
    print("="*60)
    
    # Load data
    print(f"Loading landmarks from: {LANDMARKS_FILE}")
    print(f"Loading labels from: {LABELS_FILE}")
    
    X = np.load(LANDMARKS_FILE)
    y_raw = np.load(LABELS_FILE)
    
    print(f"Landmarks shape: {X.shape}")
    print(f"Labels shape: {y_raw.shape}")
    
    # Check class distribution
    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} samples")
    
    # Filter out classes with too few samples (less than 10)
    min_samples_per_class = 10
    valid_classes = []
    for label, count in zip(unique, counts):
        if count >= min_samples_per_class:
            valid_classes.append(label)
        else:
            print(f"  ⚠️  Removing class '{label}' (only {count} samples)")
    
    # Filter data to keep only valid classes
    valid_mask = np.isin(y_raw, valid_classes)
    X = X[valid_mask]
    y_raw = y_raw[valid_mask]
    
    print(f"\nAfter filtering:")
    print(f"Landmarks shape: {X.shape}")
    print(f"Labels shape: {y_raw.shape}")
    print(f"Remaining classes: {len(np.unique(y_raw))}")
    
    # Encode string labels to integers
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print(f"Encoded labels shape: {y.shape}")
    print(f"Label mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {i}: {class_name}")
    
    return X, y, label_encoder

def split_data(X, y):
    """Split data into training and testing sets."""
    print("\n" + "="*60)
    print("  Splitting Data")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE,
        stratify=y  # Ensure balanced class distribution
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes):
    """Build and compile the neural network."""
    print("\n" + "="*60)
    print("  Building Model")
    print("="*60)
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First hidden layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with early stopping."""
    print("\n" + "="*60)
    print("  Training Model")
    print("="*60)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model performance and print metrics."""
    print("\n" + "="*60)
    print("  Evaluating Model")
    print("="*60)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy, y_pred

def save_artifacts(model, label_encoder, history):
    """Save model, encoder, and training history."""
    print("\n" + "="*60)
    print("  Saving Artifacts")
    print("="*60)
    
    # Save model
    model.save(MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    
    # Save label encoder
    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {ENCODER_FILE}")
    
    # Save training history
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    }
    
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history_dict, f)
    print("Training history saved to: models/training_history.pkl")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline."""
    print("🚀 Starting ASL Alphabet Model Training...")
    
    # Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build model
    input_shape = (X.shape[1],)  # 63 features
    num_classes = len(label_encoder.classes_)
    model = build_model(input_shape, num_classes)
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    accuracy, y_pred = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save artifacts
    save_artifacts(model, label_encoder, history)
    
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Label encoder saved: {ENCODER_FILE}")
    print("\nReady for real-time webcam testing! 🎥")

if __name__ == "__main__":
    main()
