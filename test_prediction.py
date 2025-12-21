"""
Test script to diagnose model prediction issues
"""
import pandas as pd
import numpy as np
import joblib
import os

# Load the model
model_path = 'heart_model.pkl'
print(f"Loading model from: {os.path.abspath(model_path)}")
model = joblib.load(model_path)

print(f"Model type: {type(model)}")
print(f"Model steps: {model.named_steps.keys()}")

# Test case 1: The problematic patient
print("\n" + "="*80)
print("TEST CASE 1: High-risk patient (should predict disease=1)")
print("="*80)

high_risk_patient = pd.DataFrame({
    'age': [70],
    'sex': [1],
    'cp': [3],
    'trestbps': [185],
    'chol': [390],
    'fbs': [1],
    'restecg': [2],
    'thalach': [95],
    'exang': [1],
    'oldpeak': [3.5],
    'slope': [0],
    'ca': [3],
    'thal': [3]
})

print("\nPatient Data:")
print(high_risk_patient.T)

prediction = model.predict(high_risk_patient)[0]
probabilities = model.predict_proba(high_risk_patient)[0]

print(f"\n🔍 Prediction: {prediction}")
print(f"   - 0 = No Disease")
print(f"   - 1 = Disease Detected")
print(f"\nProbabilities:")
print(f"   - No Disease (0): {probabilities[0]*100:.2f}%")
print(f"   - Disease (1): {probabilities[1]*100:.2f}%")

print(f"\n{'⚠️ ISSUE!' if prediction == 0 else '✅ CORRECT'}")

# Test case 2: A clearly healthy patient
print("\n" + "="*80)
print("TEST CASE 2: Low-risk healthy patient (should predict disease=0)")
print("="*80)

low_risk_patient = pd.DataFrame({
    'age': [30],
    'sex': [0],
    'cp': [3],  # asymptomatic
    'trestbps': [110],
    'chol': [180],
    'fbs': [0],
    'restecg': [0],
    'thalach': [180],
    'exang': [0],
    'oldpeak': [0.0],
    'slope': [0],
    'ca': [0],
    'thal': [0]
})

print("\nPatient Data:")
print(low_risk_patient.T)

prediction2 = model.predict(low_risk_patient)[0]
probabilities2 = model.predict_proba(low_risk_patient)[0]

print(f"\n🔍 Prediction: {prediction2}")
print(f"Probabilities:")
print(f"   - No Disease (0): {probabilities2[0]*100:.2f}%")
print(f"   - Disease (1): {probabilities2[1]*100:.2f}%")

print(f"\n{'✅ CORRECT' if prediction2 == 0 else '⚠️ ISSUE!'}")

# Test case 3: Load actual training data to verify
print("\n" + "="*80)
print("TEST CASE 3: Sample from original dataset")
print("="*80)

if os.path.exists('heart.csv'):
    dataset = pd.read_csv('heart.csv')
    print(f"Dataset shape: {dataset.shape}")
    print(f"\nFirst patient from dataset:")
    
    sample = dataset.head(1).drop('target', axis=1)
    actual_target = dataset.head(1)['target'].values[0]
    
    print(sample.T)
    print(f"\nActual target: {actual_target}")
    
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]
    
    print(f"Model prediction: {pred}")
    print(f"Probabilities: No Disease={prob[0]*100:.2f}%, Disease={prob[1]*100:.2f}%")
    print(f"\n{'✅ MATCH' if pred == actual_target else '⚠️ MISMATCH'}")
    
    # Test a patient with target=1 (has disease)
    print("\n" + "="*80)
    print("TEST CASE 4: Patient with confirmed disease (target=1)")
    print("="*80)
    
    disease_patient = dataset[dataset['target'] == 1].head(1).drop('target', axis=1)
    actual_target2 = 1
    
    print(disease_patient.T)
    
    pred2 = model.predict(disease_patient)[0]
    prob2 = model.predict_proba(disease_patient)[0]
    
    print(f"\nActual target: {actual_target2}")
    print(f"Model prediction: {pred2}")
    print(f"Probabilities: No Disease={prob2[0]*100:.2f}%, Disease={prob2[1]*100:.2f}%")
    print(f"\n{'✅ MATCH' if pred2 == actual_target2 else '⚠️ MISMATCH'}")
else:
    print("heart.csv not found - skipping dataset verification")

# Model introspection
print("\n" + "="*80)
print("MODEL INTROSPECTION")
print("="*80)

print(f"\nModel class: {model.named_steps['model'].__class__.__name__}")
print(f"Number of estimators: {model.named_steps['model'].n_estimators if hasattr(model.named_steps['model'], 'n_estimators') else 'N/A'}")

# Check if SMOTE is in the pipeline
if 'smote' in model.named_steps:
    print("✅ SMOTE is in the pipeline")
else:
    print("⚠️ SMOTE not found in pipeline")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
