"""
Quick test: Retrain model with proper verification
This script retrains a simple model to verify the issue
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
print("Loading data...")
df = pd.read_csv('heart.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nTarget values: {y.unique()}")
print(f"  - 0 (No Disease): {(y==0).sum()} patients")
print(f"  - 1 (Disease): {(y==1).sum()} patients")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining simple Random Forest...")
# Train a simple model (without pipeline for testing)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test on training data
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Test with your problematic patient
print("\n" + "="*80)
print("Testing HIGH-RISK patient (should predict 1 = Disease)")
print("="*80)

high_risk = pd.DataFrame({
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

pred = clf.predict(high_risk)[0]
proba = clf.predict_proba(high_risk)[0]

print(f"\nPrediction: {pred}")
print(f"Probabilities: No Disease={proba[0]*100:.2f}%, Disease={proba[1]*100:.2f}%")
print(f"\nResult: {'✅ CORRECT (predicted disease)' if pred == 1 else '⚠️ STILL WRONG (predicted no disease)'}")

# Test with healthy patient
print("\n" + "="*80)
print("Testing LOW-RISK patient (should predict 0 = No Disease)")
print("="*80)

low_risk = pd.DataFrame({
    'age': [30],
    'sex': [0],
    'cp': [3],
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

pred2 = clf.predict(low_risk)[0]
proba2 = clf.predict_proba(low_risk)[0]

print(f"\nPrediction: {pred2}")
print(f"Probabilities: No Disease={proba2[0]*100:.2f}%, Disease={proba2[1]*100:.2f}%")
print(f"\nResult: {'✅ CORRECT (predicted no disease)' if pred2 == 0 else '⚠️ WRONG (predicted disease)'}")

# Analyze feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Top 5)")
print("="*80)
importances = clf.feature_importances_
feature_names = X.columns
indices = importances.argsort()[::-1][:5]

for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If this simple model predicts correctly, the issue is with the")
print("complex pipeline model (heart_model.pkl). You need to re-run")
print("the entire Jupyter notebook to regenerate it properly.")
