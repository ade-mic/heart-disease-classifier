#!/usr/bin/env python3
"""
Helper script to setup the model file for the FastAPI app
"""

import os
import shutil
import sys

# Possible source locations for the model
SOURCE_PATHS = [
    "/content/drive/MyDrive/Colab Notebooks/heart_model.pkl",  # Google Drive
    "../heart_model.pkl",
    "../../heart_model.pkl",
    "./heart_model.pkl",
]

# Target location
TARGET_PATH = "../heart_model.pkl"


def find_model():
    """Find the model file"""
    print("🔍 Searching for model file...")
    
    for path in SOURCE_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"✅ Found model at: {abs_path}")
            return abs_path
    
    return None


def setup_model():
    """Setup the model file"""
    print("=" * 60)
    print("Heart Disease Model Setup")
    print("=" * 60)
    
    # Check if target already exists
    target_abs = os.path.abspath(TARGET_PATH)
    if os.path.exists(target_abs):
        size_mb = os.path.getsize(target_abs) / (1024 * 1024)
        print(f"\n✅ Model already exists at: {target_abs}")
        print(f"   Size: {size_mb:.2f} MB")
        print("\n✨ Setup complete! You can now run the FastAPI app.")
        return True
    
    # Find source model
    source = find_model()
    
    if source is None:
        print("\n❌ Model file not found!")
        print("\n📋 To fix this:")
        print("1. Run the Jupyter notebook 'Group_Assignment__1.ipynb'")
        print("2. The notebook will generate 'heart_model.pkl'")
        print("3. Make sure the model is saved in one of these locations:")
        for path in SOURCE_PATHS:
            print(f"   - {os.path.abspath(path)}")
        print(f"\n   Or copy it manually to: {target_abs}")
        return False
    
    # Copy model to target
    try:
        print(f"\n📦 Copying model to: {target_abs}")
        shutil.copy2(source, target_abs)
        size_mb = os.path.getsize(target_abs) / (1024 * 1024)
        print(f"✅ Model copied successfully! Size: {size_mb:.2f} MB")
        print("\n✨ Setup complete! You can now run the FastAPI app.")
        return True
    except Exception as e:
        print(f"\n❌ Error copying model: {e}")
        return False


if __name__ == "__main__":
    success = setup_model()
    sys.exit(0 if success else 1)
