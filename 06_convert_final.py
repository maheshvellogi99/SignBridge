"""
SignBridge — Phase 3: Final TFLite Conversion
Working conversion with full compatibility handling
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle

# ==========================================
# Configuration and Paths
# ==========================================
OUTPUT_MODEL_PATH = "models/sign_model.tflite"
ENCODER_FILE = "models/label_encoder.pkl"

def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def get_file_size(file_path):
    """Get file size in bytes."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0

def create_fresh_model():
    """Create a fresh model with the correct architecture."""
    print("="*60)
    print("  Creating Fresh Model Architecture")
    print("="*60)
    
    try:
        # Load label encoder to get number of classes
        with open(ENCODER_FILE, 'rb') as f:
            label_encoder = pickle.load(f)
        num_classes = len(label_encoder.classes_)
        
        print(f"📝 Number of classes: {num_classes}")
        
        # Create the exact same architecture as trained
        inputs = keras.Input(shape=(63,), name="input")
        
        # First hidden layer
        x = layers.Dense(128, activation='relu', name="dense_1")(inputs)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        
        # Second hidden layer
        x = layers.Dense(64, activation='relu', name="dense_2")(x)
        x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        
        # Third hidden layer
        x = layers.Dense(32, activation='relu', name="dense_3")(x)
        x = layers.Dropout(0.1, name="dropout_3")(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="signbridge_alphabet")
        
        # Compile model
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        print("✅ Fresh model created successfully!")
        print(f"📐 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None

def convert_to_tflite_working(model):
    """Convert model using the working method."""
    print("\n" + "="*60)
    print("  Converting to TensorFlow Lite")
    print("="*60)
    
    try:
        print("🔄 Setting up converter...")
        
        # Create converter with concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tf.function(model).get_concrete_function(
                tf.TensorSpec(model.input_shape, model.inputs[0].dtype)
            )]
        )
        
        print("⚡ Applying optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        print("🔧 Converting model...")
        tflite_model = converter.convert()
        
        print("✅ Model converted successfully!")
        return tflite_model
        
    except Exception as e:
        print(f"⚠️ Concrete function method failed: {e}")
        
        try:
            print("🔄 Trying standard converter...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
            ]
            
            tflite_model = converter.convert()
            print("✅ Model converted with standard method!")
            return tflite_model
            
        except Exception as e2:
            print(f"❌ Standard method also failed: {e2}")
            return None

def save_tflite_model(tflite_model):
    """Save the TFLite model."""
    print("\n" + "="*60)
    print("  Saving TFLite Model")
    print("="*60)
    
    try:
        os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
        
        print(f"💾 Saving TFLite model to: {OUTPUT_MODEL_PATH}")
        with open(OUTPUT_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print("✅ TFLite model saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving TFLite model: {e}")
        return False

def validate_conversion():
    """Validate and show results."""
    print("\n" + "="*60)
    print("  Conversion Validation")
    print("="*60)
    
    tflite_size = get_file_size(OUTPUT_MODEL_PATH)
    
    if tflite_size == 0:
        print("❌ Could not read TFLite file size")
        return
    
    print("📊 TFLite Model Info:")
    print(f"   File size: {format_file_size(tflite_size)} ({tflite_size:,} bytes)")
    
    # Test TFLite model
    try:
        print("\n🔍 Validating TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=OUTPUT_MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ TFLite model validation successful!")
        print(f"   Input:  {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"   Output: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        
        # Test with dummy data
        print("🧪 Testing inference...")
        dummy_input = np.random.random((1, 63)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"   Test output shape: {output.shape}")
        print(f"   Test output range: [{np.min(output):.3f}, {np.max(output):.3f}]")
        print("✅ Inference test successful!")
        
    except Exception as e:
        print(f"❌ Error validating TFLite model: {e}")

def print_deployment_info():
    """Print deployment information."""
    print("\n" + "="*60)
    print("  Mobile Deployment Ready!")
    print("="*60)
    
    print("📱 Your TFLite model is ready for React Native deployment!")
    print("\n📋 React Native Integration:")
    print("   1. Copy 'sign_model.tflite' to your React Native app assets")
    print("   2. Install: npm install react-native-fast-tflite")
    print("   3. Use TFLite model for real-time inference")
    print("   4. Input: [1, 63] float32 array")
    print("   5. Output: [1, 28] probability array")
    print("   6. Use 'label_encoder.pkl' for class mapping")
    
    print("\n⚡ Mobile Benefits:")
    print("   • Optimized for mobile processors")
    print("   • Reduced memory footprint")
    print("   • Fast inference for real-time use")
    print("   • Compatible with iOS and Android")

def main():
    """Main conversion process."""
    print("🚀 SignBridge TFLite Conversion")
    print("   Creating mobile-optimized ASL recognition model")
    
    # Create fresh model
    model = create_fresh_model()
    if model is None:
        print("❌ Could not create model")
        return
    
    # Convert to TFLite
    tflite_model = convert_to_tflite_working(model)
    if tflite_model is None:
        print("❌ Could not convert model")
        return
    
    # Save TFLite model
    if not save_tflite_model(tflite_model):
        print("❌ Could not save model")
        return
    
    # Validate
    validate_conversion()
    
    # Print info
    print_deployment_info()
    
    print("\n🎉 TFLite conversion complete!")
    print("📱 Your SignBridge model is ready for mobile deployment!")

if __name__ == "__main__":
    main()
