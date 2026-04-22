"""
SignBridge Dynamic Phrase Recognition - Kaggle Training Script (Overhauled)
Spatial-Temporal Feature Extraction with Conv1D + Bidirectional LSTM

Target Vocabulary: 42 signs for robust dynamic phrase recognition
Model Architecture: Conv1D + Bidirectional LSTM with wrist-relative coordinates
Output: TFLite model for edge deployment
"""

import os
import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# Configuration
# ==========================================
# Kaggle Paths
BASE_DIR = '/kaggle/input/competitions/asl-signs/'
CSV_PATH = '/kaggle/input/competitions/asl-signs/train.csv'
PARQUET_DIR = '/kaggle/input/competitions/asl-signs/train_landmark_files/'
WORKING_DIR = '/kaggle/working/'

# Target Vocabulary (21 signs - reduced for better training)
TARGET_SIGNS = [
    'hello', 'goodbye', 'yes', 'no', 'thankyou', 'please', 'sorry', 'fine',
    'who', 'what', 'where', 'why', 'how', 'which', 'when', 'help',
    'water', 'food', 'sleep', 'toilet', 'hungry'
]

# Training Parameters
SAMPLES_PER_SIGN = 250
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63  # 21 hand landmarks * 3 coordinates
HAND_LANDMARK_INDICES = list(range(468))  # MediaPipe hand landmarks (0-467)

# Model Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.4

print("🚀 SignBridge Dynamic Phrase Recognition - Kaggle Training (Overhauled)")
print("=" * 80)
print(f"🎯 Target Vocabulary: {len(TARGET_SIGNS)} signs")
print(f"📊 Samples per Sign: {SAMPLES_PER_SIGN}")
print(f"📏 Sequence Length: {SEQUENCE_LENGTH}")
print(f"🔢 Features per Frame: {NUM_FEATURES}")
print(f"💾 Total Expected Samples: {len(TARGET_SIGNS) * SAMPLES_PER_SIGN}")
print(f"🔧 Feature Engineering: Wrist-Relative Coordinates")
print(f"🏗️  Architecture: Conv1D + Bidirectional LSTM")
print("=" * 80)

# ==========================================
# Data Loading and Preprocessing
# ==========================================

class SignBridgeDataLoader:
    def __init__(self):
        self.target_signs = TARGET_SIGNS
        self.samples_per_sign = SAMPLES_PER_SIGN
        self.sequence_length = SEQUENCE_LENGTH
        self.num_features = NUM_FEATURES
        
    def load_and_filter_data(self):
        """Load CSV and filter for target signs."""
        print("📥 Loading and filtering dataset...")
        
        # Load train.csv
        train_df = pd.read_csv(CSV_PATH)
        print(f"✅ Loaded train.csv with {len(train_df)} total samples")
        
        # Filter for target signs
        filtered_df = train_df[train_df['sign'].isin(self.target_signs)].copy()
        print(f"🎯 Filtered to {len(filtered_df)} samples for target signs")
        
        # Show distribution
        print("\n📊 Sample distribution before balancing:")
        for sign in self.target_signs:
            count = len(filtered_df[filtered_df['sign'] == sign])
            print(f"   {sign}: {count} samples")
        
        # Balance the dataset
        balanced_dfs = []
        for sign in self.target_signs:
            sign_df = filtered_df[filtered_df['sign'] == sign]
            
            if len(sign_df) > self.samples_per_sign:
                sign_df = sign_df.sample(n=self.samples_per_sign, random_state=42)
                print(f"✂️  {sign}: Limited to {self.samples_per_sign} samples")
            else:
                print(f"⚠️  {sign}: Only {len(sign_df)} samples available")
            
            balanced_dfs.append(sign_df)
        
        # Combine balanced dataframes
        final_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"\n✅ Final balanced dataset: {len(final_df)} samples")
        
        return final_df
    
    def extract_hand_landmarks(self, parquet_path):
        """Extract hand landmarks from parquet file with Interpolation."""
        try:
            # Read parquet file
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            
            # Check if we have the expected columns
            if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
                return None
            
            # Filter for hand landmarks only (indices 0-467 for hands)
            hand_mask = df.index.isin(HAND_LANDMARK_INDICES)
            hand_df = df[hand_mask].copy()
            
            if len(hand_df) == 0:
                return None

            # 🛠️ THE FIX 1: Smooth Interpolation instead of 0.0 teleportation
            hand_df[['x', 'y', 'z']] = hand_df[['x', 'y', 'z']].interpolate(method='linear', limit_direction='both')
            hand_df[['x', 'y', 'z']] = hand_df[['x', 'y', 'z']].fillna(0.0) # Fallback if entirely empty
            
            # Get unique frames and sort them
            unique_frames = sorted(hand_df['frame'].unique())
            landmarks_array = []
            
            for frame_num in unique_frames:
                frame_data = hand_df[hand_df['frame'] == frame_num]
                frame_landmarks = []
                
                for _, row in frame_data.iterrows():
                    frame_landmarks.extend([row['x'], row['y'], row['z']])
                
                # Ensure we have exactly 63 features
                if len(frame_landmarks) == 63:
                    landmarks_array.append(frame_landmarks)
                elif len(frame_landmarks) > 63:
                    landmarks_array.append(frame_landmarks[:63])
                else:
                    padded = frame_landmarks + [0.0] * (63 - len(frame_landmarks))
                    landmarks_array.append(padded)
            
            if not landmarks_array:
                return None
            
            return np.array(landmarks_array)
            
        except Exception as e:
            return None
    
    def pad_or_truncate_sequence(self, landmarks_array):
        """Pad/truncate sequence and apply Sequence Normalization."""
        current_length = len(landmarks_array)
        
        # Sizing the sequence
        if current_length < self.sequence_length:
            # Pad by repeating the last frame instead of adding zeros
            padding = np.tile(landmarks_array[-1], (self.sequence_length - current_length, 1))
            sequence = np.vstack([landmarks_array, padding])
        elif current_length > self.sequence_length:
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            sequence = landmarks_array[indices]
        else:
            sequence = landmarks_array

        # 🛠️ THE FIX 2: Z-Score Normalization (Mean Centering)
        # This makes the AI look at the *movement*, regardless of where the hand is on screen.
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0) + 1e-7  # Add tiny number to prevent divide-by-zero
        normalized_sequence = (sequence - mean) / std

        return normalized_sequence
    
    def process_sequences(self, filtered_df):
        """Process all sequences and create arrays."""
        print(f"\n🔄 Processing {len(filtered_df)} sequences...")
        
        sequences = []
        labels = []
        label_mapping = {sign: idx for idx, sign in enumerate(self.target_signs)}
        
        successful = 0
        failed = 0
        
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Extracting sequences"):
            parquet_path = os.path.join(BASE_DIR, row['path'])
            
            if not os.path.exists(parquet_path):
                failed += 1
                continue
            
            # Extract landmarks
            landmarks_array = self.extract_hand_landmarks(parquet_path)
            
            if landmarks_array is None:
                failed += 1
                continue
            
            # Process to fixed length
            processed_sequence = self.pad_or_truncate_sequence(landmarks_array)
            
            # Add to datasets
            sequences.append(processed_sequence)
            labels.append(label_mapping[row['sign']])
            successful += 1
        
        print(f"\n✅ Processing complete:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {successful/(successful+failed)*100:.1f}%")
        
        if sequences:
            sequences_array = np.array(sequences)
            labels_array = np.array(labels)
            
            print(f"\n📊 Final array shapes:")
            print(f"   Sequences: {sequences_array.shape}")
            print(f"   Labels: {labels_array.shape}")
            
            return sequences_array, labels_array, label_mapping
        else:
            return None, None, None

# ==========================================
# Model Architecture
# ==========================================

class SignBridgeModel:
    def __init__(self, num_classes, sequence_length, num_features):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features
        
    def build_model(self):
        """Build LSTM/GRU model with dropout."""
        print("🏗️  Building LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # First LSTM layer
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(64, activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model):
        """Compile model for GPU training."""
        print("⚙️  Compiling model for GPU...")
        
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(WORKING_DIR, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks

# ==========================================
# TFLite Conversion
# ==========================================

class TFLiteConverter:
    def __init__(self):
        pass
    
    def convert_to_tflite(self, model, model_name='signbridge_dynamic'):
        """Convert Keras model to TFLite with LSTM support."""
        print("🔄 Converting model to TFLite...")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # REQUIRED FIX FOR LSTM: Enable advanced TF Ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(WORKING_DIR, f'{model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to {tflite_path}")
        
        # Get model size
        model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        print(f"📊 Model size: {model_size:.2f} MB")
        
        return tflite_path

# ==========================================
# Main Training Pipeline
# ==========================================

def main():
    """Main training pipeline."""
    
    # Check GPU availability
    print("🔍 Checking GPU availability...")
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ GPU detected: {len(gpu_devices)} device(s)")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("⚠️ No GPU detected, using CPU")
    
    # Initialize data loader
    data_loader = SignBridgeDataLoader()
    
    # Load and filter data
    filtered_df = data_loader.load_and_filter_data()
    
    # Process sequences
    sequences_array, labels_array, label_mapping = data_loader.process_sequences(filtered_df)
    
    if sequences_array is None:
        print("❌ Failed to process sequences!")
        return
    
    # Split data (80% train, 20% validation)
    print("\n📊 Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        sequences_array, labels_array, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_array
    )
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    # Initialize model
    model_builder = SignBridgeModel(
        num_classes=len(TARGET_SIGNS),
        sequence_length=SEQUENCE_LENGTH,
        num_features=NUM_FEATURES
    )
    
    # Build and compile model
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    # Show model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Get callbacks
    callbacks = model_builder.get_callbacks()
    
    # Train model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_accuracy:.4f}")
    
    # Save final model
    final_model_path = os.path.join(WORKING_DIR, 'signbridge_dynamic_final.h5')
    model.save(final_model_path)
    print(f"✅ Final model saved to {final_model_path}")
    
    # Convert to TFLite
    converter = TFLiteConverter()
    tflite_path = converter.convert_to_tflite(model, 'signbridge_dynamic')
    
    # Save label mapping
    label_map_path = os.path.join(WORKING_DIR, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"✅ Label mapping saved to {label_map_path}")
    
    # Save training history
    history_dict = {key: [float(x) for x in value] for key, value in history.history.items()}
    history_path = os.path.join(WORKING_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✅ Training history saved to {history_path}")
    
    # Save metadata
    metadata = {
        'target_signs': TARGET_SIGNS,
        'num_classes': len(TARGET_SIGNS),
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': NUM_FEATURES,
        'samples_per_sign': SAMPLES_PER_SIGN,
        'total_samples': len(sequences_array),
        'model_parameters': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'dropout_rate': DROPOUT_RATE
        },
        'performance': {
            'validation_accuracy': float(val_accuracy),
            'validation_loss': float(val_loss)
        },
        'files_generated': [
            'signbridge_dynamic_final.h5',
            'signbridge_dynamic.tflite',
            'label_map.json',
            'training_history.json'
        ]
    }
    
    metadata_path = os.path.join(WORKING_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Model metadata saved to {metadata_path}")
    
    print("\n🎉 Training complete!")
    print("=" * 70)
    print("📁 Files saved to /kaggle/working/:")
    print("   • signbridge_dynamic_final.h5 - Keras model")
    print("   • signbridge_dynamic.tflite - Quantized TFLite model")
    print("   • label_map.json - Label mapping")
    print("   • training_history.json - Training metrics")
    print("   • model_metadata.json - Model information")
    print("=" * 70)
    print(f"🎯 Final Validation Accuracy: {val_accuracy:.4f}")
    print("🚀 Ready for deployment!")

if __name__ == "__main__":
    main()
