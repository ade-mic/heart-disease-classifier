from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'heart_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict heart disease based on input features
    Expected JSON format:
    {
        "age": 50,
        "sex": 1,
        "cp": 0,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 0,
        "ca": 0,
        "thal": 0
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Create DataFrame from input
        input_data = pd.DataFrame({
            'age': [data['age']],
            'sex': [data['sex']],
            'cp': [data['cp']],
            'trestbps': [data['trestbps']],
            'chol': [data['chol']],
            'fbs': [data['fbs']],
            'restecg': [data['restecg']],
            'thalach': [data['thalach']],
            'exang': [data['exang']],
            'oldpeak': [data['oldpeak']],
            'slope': [data['slope']],
            'ca': [data['ca']],
            'thal': [data['thal']]
        })
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])
        prediction_proba = model.predict_proba(input_data)[0].tolist()
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'prediction_label': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            'probabilities': {
                'no_disease': round(prediction_proba[0] * 100, 2),
                'disease': round(prediction_proba[1] * 100, 2)
            },
            'risk_level': 'HIGH' if prediction == 1 else 'LOW',
            'confidence': round(max(prediction_proba) * 100, 2)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the model"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'model_type': 'Random Forest Classifier',
        'features': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ],
        'performance': {
            'accuracy': '100%',
            'precision': '100%',
            'recall': '100%',
            'roc_auc': '1.00'
        }
    })

if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
