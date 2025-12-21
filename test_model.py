import joblib
import pandas as pd
import sys

try:
    print("🔍 Loading model...")
    model = joblib.load('heart_model.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    test_data = pd.DataFrame({
        'age': [50],
        'sex': [1],
        'cp': [0],
        'trestbps': [120],
        'chol': [200],
        'fbs': [0],
        'restecg': [0],
        'thalach': [150],
        'exang': [0],
        'oldpeak': [0.0],
        'slope': [0],
        'ca': [0],
        'thal': [0]
    })
    
    prediction = model.predict(test_data)[0]
    proba = model.predict_proba(test_data)[0]
    
    print(f"✅ Prediction: {prediction}")
    print(f"   Probabilities: No Disease={proba[0]:.2%}, Disease={proba[1]:.2%}")
    print("\n🎉 Everything works perfectly!")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
