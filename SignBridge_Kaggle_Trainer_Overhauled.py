"""
SignBridge Dynamic Phrase Recognition - Kaggle Training Script (Overhauled)
Spatial-Temporal Feature Extraction with Conv1D + Bidirectional LSTM

Target Vocabulary: 21 signs for robust dynamic phrase recognition
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
        """Extract hand landmarks with wrist-relative coordinates."""
        try:
            # Read parquet file
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            
            # Check if we have the expected columns
            if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
                return None
            
            # Filter for hand landmarks only (indices 0-467 for hands)
            hand_mask = df.index.isin(HAND_LANDMARK_INDICES)
            hand_df = df[hand_mask]
            
            if len(hand_df) == 0:
                return None
            
            # Get unique frames and sort them
            unique_frames = sorted(hand_df['frame'].unique())
            
            # Create landmarks array
            landmarks_array = []
            
            for frame_num in unique_frames:
                frame_data = hand_df[hand_df['frame'] == frame_num]
                
                # Extract landmarks for this frame
                frame_landmarks = []
                
                # Get all landmarks in this frame
                for _, row in frame_data.iterrows():
                    frame_landmarks.extend([row['x'], row['y'], row['z']])
                
                # Ensure we have exactly 63 features (21 landmarks * 3 coords)
                if len(frame_landmarks) < 63:
                    # Pad with zeros if not enough landmarks
                    frame_landmarks.extend([0.0] * (63 - len(frame_landmarks)))
                elif len(frame_landmarks) > 63:
                    # Truncate if too many landmarks
                    frame_landmarks = frame_landmarks[:63]
                
                # Convert to numpy array for processing
                frame_array = np.array(frame_landmarks)
                
                # SPATIAL FEATURE ENGINEERING: Convert to wrist-relative coordinates
                if len(frame_array) >= 3:  # Ensure we have at least wrist coordinates
                    # Extract wrist coordinates (landmark 0: indices 0,1,2)
                    wrist_x = frame_array[0]
                    wrist_y = frame_array[1]
                    wrist_z = frame_array[2]
                    
                    # Create relative coordinates array
                    relative_landmarks = []
                    
                    # Wrist becomes origin (0,0,0)
                    relative_landmarks.extend([0.0, 0.0, 0.0])
                    
                    # Subtract wrist coordinates from all other landmarks
                    for i in range(3, len(frame_array), 3):
                        if i + 2 < len(frame_array):
                            rel_x = frame_array[i] - wrist_x
                            rel_y = frame_array[i + 1] - wrist_y
                            rel_z = frame_array[i + 2] - wrist_z
                            relative_landmarks.extend([rel_x, rel_y, rel_z])
                    
                    # Ensure we still have 63 features
                    if len(relative_landmarks) < 63:
                        relative_landmarks.extend([0.0] * (63 - len(relative_landmarks)))
                    elif len(relative_landmarks) > 63:
                        relative_landmarks = relative_landmarks[:63]
                    
                    frame_array = np.array(relative_landmarks)
                
                landmarks_array.append(frame_array)
            
            if not landmarks_array:
                return None
            
            return np.array(landmarks_array)
            
        except Exception as e:
            return None
    
    def pad_or_truncate_sequence(self, landmarks_array):
        """Pad or truncate sequence to fixed length."""
        current_length = len(landmarks_array)
        
        if current_length == self.sequence_length:
            return landmarks_array
        elif current_length < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - current_length, self.num_features))
            return np.vstack([landmarks_array, padding])
        else:
            # Truncate by sampling evenly
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            return landmarks_array[indices]
    
    def process_sequences(self, filtered_df):
        """Process all sequences and create arrays."""
        print(f"\n🔄 Processing {len(filtered_df)} sequences...")
        
        sequences = []
        labels = []
        label_mapping = {sign: idx for idx, sign in enumerate(self.target_signs)}
        
        successful = 0
        failed = 0
        
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Extracting sequences"):
            parquet_path = os.path.join(PARQUET_DIR, f"{row['path']}.parquet")
            
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
# Model Architecture (Overhauled)
# ==========================================

class SignBridgeModel:
    def __init__(self, num_classes, sequence_length, num_features):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features
        
    def build_model(self):
        """Build Conv1D + Bidirectional LSTM model."""
        print("🏗️  Building Conv1D + Bidirectional LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # SPATIAL FEATURE EXTRACTION: Conv1D layer
            layers.Conv1D(
                filters=64, 
                kernel_size=3, 
                activation='relu', 
                padding='same',
                name='conv1d_spatial'
            ),
            layers.MaxPooling1D(pool_size=2, name='maxpool_spatial'),
            layers.BatchNormalization(name='bn_conv'),
            
            # TEMPORAL FEATURE EXTRACTION: Bidirectional LSTM 1
            layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
                name='bilstm_1'
            ),
            layers.BatchNormalization(name='bn_lstm1'),
            
            # TEMPORAL FEATURE EXTRACTION: Bidirectional LSTM 2
            layers.Bidirectional(
                layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
                name='bilstm_2'
            ),
            layers.BatchNormalization(name='bn_lstm2'),
            
            # DENSE LAYERS with Dropout
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(DROPOUT_RATE, name='dropout_1'),
            
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.BatchNormalization(name='bn_dense2'),
            layers.Dropout(DROPOUT_RATE, name='dropout_2'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
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
        """Get training callbacks with enhanced patience."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience for better training
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
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
# TFLite Conversion (Enhanced)
# ==========================================

class TFLiteConverter:
    def __init__(self):
        pass
    
    def convert_to_tflite(self, model, model_name='signbridge_dynamic'):
        """Convert Keras model to TFLite with quantization."""
        print("🔄 Converting model to TFLite...")
        
        # Enable SELECT_TF_OPS for compatibility
        try:
            from tensorflow.lite.python.optimizer import calibrator
        except ImportError:
            print("⚠️ SELECT_TF_OPS not available, using standard conversion")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable SELECT_TF_OPS for complex operations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
        ]
        
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
        'feature_engineering': 'wrist_relative_coordinates',
        'architecture': 'conv1d_bidirectional_lstm',
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
    print("=" * 80)
    print("📁 Files saved to /kaggle/working/:")
    print("   • signbridge_dynamic_final.h5 - Keras model")
    print("   • signbridge_dynamic.tflite - Quantized TFLite model")
    print("   • label_map.json - Label mapping")
    print("   • training_history.json - Training metrics")
    print("   • model_metadata.json - Model information")
    print("=" * 80)
    print(f"🎯 Final Validation Accuracy: {val_accuracy:.4f}")
    print("🚀 Ready for deployment!")

if __name__ == "__main__":
    main()
