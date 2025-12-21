# Export the model from the notebook to the project root
import joblib
import sys

# Load model from Google Drive path (update if different)
model_path = "/content/drive/MyDrive/Colab Notebooks/heart_model.pkl"
output_path = "./heart_model.pkl"

try:
    model = joblib.load(model_path)
    joblib.dump(model, output_path)
    print(f"✅ Model successfully exported to {output_path}")
except FileNotFoundError:
    print(f"❌ Model not found at {model_path}")
    print("Please update the path or run the notebook to generate the model first.")
    sys.exit(1)
