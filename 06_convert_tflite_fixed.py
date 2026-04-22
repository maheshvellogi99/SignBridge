"""
SignBridge — Phase 3: Edge AI Conversion (Fixed)
TensorFlow Lite Model Conversion for Mobile Deployment

This script handles version compatibility issues and converts the trained 
Keras model into a mobile-optimized TensorFlow Lite model.
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# ==========================================
# Configuration and Paths
# ==========================================
INPUT_MODEL_PATH = "models/alphabet_model.h5"
OUTPUT_MODEL_PATH = "models/sign_model.tflite"
CLEAN_MODEL_PATH = "models/alphabet_model_clean.h5"

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

def clean_and_save_model():
    """Load the model and re-save it to remove compatibility issues."""
    print("="*60)
    print("  Cleaning Model for Compatibility")
    print("="*60)
    
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"❌ Error: Input model not found at {INPUT_MODEL_PATH}")
        return None
    
    try:
        print("🔄 Loading model with legacy compatibility...")
        
        # Set TensorFlow legacy behavior
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        
        # Try loading with legacy Keras
        try:
            # First try with the new method
            model = keras.models.load_model(INPUT_MODEL_PATH, compile=False)
        except:
            # If that fails, try with legacy loading
            print("🔄 Trying legacy Keras loading...")
            from tensorflow.keras.models import load_model
            model = load_model(INPUT_MODEL_PATH, compile=False)
        
        print("✅ Model loaded successfully!")
        
        # Re-save the model to clean up compatibility issues
        print("💾 Re-saving model to clean compatibility issues...")
        model.save(CLEAN_MODEL_PATH, save_format='h5')
        print("✅ Model cleaned and saved successfully!")
        
        # Display model info
        print(f"\n📋 Model Architecture:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"❌ Error cleaning model: {e}")
        return None

def convert_to_tflite_simple(model):
    """Convert model to TFLite with basic approach."""
    print("\n" + "="*60)
    print("  Converting to TensorFlow Lite")
    print("="*60)
    
    try:
        print("🔄 Initializing TFLite converter...")
        
        # Create a simple converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply basic optimizations
        print("⚡ Applying optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        print("🔧 Converting model...")
        tflite_model = converter.convert()
        
        print("✅ Model converted successfully!")
        return tflite_model
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return None

def save_tflite_model(tflite_model):
    """Save the TFLite model to disk."""
    print("\n" + "="*60)
    print("  Saving TFLite Model")
    print("="*60)
    
    try:
        # Ensure models directory exists
        os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
        
        # Save TFLite model
        print(f"💾 Saving TFLite model to: {OUTPUT_MODEL_PATH}")
        with open(OUTPUT_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print("✅ TFLite model saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving TFLite model: {e}")
        return False

def validate_conversion():
    """Validate the conversion and show compression statistics."""
    print("\n" + "="*60)
    print("  Conversion Validation")
    print("="*60)
    
    # Get file sizes
    original_size = get_file_size(INPUT_MODEL_PATH)
    tflite_size = get_file_size(OUTPUT_MODEL_PATH)
    
    if original_size == 0:
        print(f"❌ Could not read original model size")
        return
    
    if tflite_size == 0:
        print(f"❌ Could not read TFLite model size")
        return
    
    # Calculate compression ratio
    compression_ratio = (1 - tflite_size / original_size) * 100
    
    print("📊 File Size Comparison:")
    print(f"   Original Keras Model: {format_file_size(original_size)} ({original_size:,} bytes)")
    print(f"   TFLite Model:        {format_file_size(tflite_size)} ({tflite_size:,} bytes)")
    print(f"   Compression:         {compression_ratio:.1f}% reduction")
    print(f"   Size Ratio:           {tflite_size/original_size:.3f}x")
    
    # Validate model can be loaded
    try:
        print("\n🔍 Validating TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=OUTPUT_MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ TFLite model validation successful!")
        print(f"   Input Details:  {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"   Output Details: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        
    except Exception as e:
        print(f"❌ Error validating TFLite model: {e}")

def print_mobile_deployment_info():
    """Print information about mobile deployment."""
    print("\n" + "="*60)
    print("  Mobile Deployment Ready!")
    print("="*60)
    
    print("📱 Your TFLite model is ready for React Native deployment!")
    print("\n📋 Next Steps for Mobile Integration:")
    print("   1. Copy 'sign_model.tflite' to your React Native app assets")
    print("   2. Use 'react-native-fast-tflite' for inference")
    print("   3. Input shape: [1, 63] (batch_size, features)")
    print("   4. Output shape: [1, 28] (batch_size, classes)")
    print("   5. Remember to include 'label_encoder.pkl' for class mapping")
    
    print("\n⚡ Performance Benefits:")
    print("   • Reduced model size for faster app downloads")
    print("   • Optimized for mobile CPU/GPU execution")
    print("   • Lower memory footprint on device")
    print("   • Faster inference times on mobile hardware")

def main():
    """Main conversion pipeline."""
    print("🚀 Starting TensorFlow Lite Conversion (Fixed)...")
    print("   Converting ASL Alphabet model for mobile deployment")
    
    # Clean and load model
    model = clean_and_save_model()
    if model is None:
        return
    
    # Convert to TFLite
    tflite_model = convert_to_tflite_simple(model)
    if tflite_model is None:
        return
    
    # Save TFLite model
    if not save_tflite_model(tflite_model):
        return
    
    # Validate conversion
    validate_conversion()
    
    # Print deployment info
    print_mobile_deployment_info()
    
    print("\n🎉 Conversion complete! Your model is ready for mobile deployment!")
    
    # Clean up temporary file
    if os.path.exists(CLEAN_MODEL_PATH):
        os.remove(CLEAN_MODEL_PATH)
        print("🧹 Temporary cleaned model removed")

if __name__ == "__main__":
    main()
