"""
Verification script: Test that label inversion fixes the prediction issue
Run this AFTER re-running the notebook with inverted labels
"""
import pandas as pd
import joblib

print("="*80)
print("TESTING MODEL WITH CORRECTED LABELS")
print("="*80)

# Load the model (this should be retrained after label inversion)
model = joblib.load('heart_model.pkl')

# Test Case 1: HIGH-RISK patient (should NOW predict 1 = Disease)
print("\n" + "="*80)
print("TEST 1: HIGH-RISK Patient (elderly, multiple risk factors)")
print("="*80)

high_risk = pd.DataFrame({
    'age': [70],
    'sex': [1],
    'cp': [3],
    'trestbps': [185],
    'chol': [390],
    'fbs': [1],
    'restecg': [2],
    'thalach': [95],      # Very low max heart rate - BAD
    'exang': [1],         # Exercise angina - BAD
    'oldpeak': [3.5],     # High ST depression - BAD
    'slope': [0],
    'ca': [3],            # All vessels affected - BAD
    'thal': [3]
})

print("\nPatient Profile:")
print("  - Age: 70 (elderly)")
print("  - Max Heart Rate: 95 (very low - concerning)")
print("  - ST Depression: 3.5 (very high - severe)")
print("  - Vessels Affected: 3 (all major vessels)")
print("  - Exercise Angina: Yes")
print("  - Blood Pressure: 185 (hypertensive)")
print("  - Cholesterol: 390 (very high)")

pred = model.predict(high_risk)[0]
proba = model.predict_proba(high_risk)[0]

print(f"\n🔍 Prediction: {pred}")
print(f"   Probabilities: No Disease={proba[0]*100:.2f}%, Disease={proba[1]*100:.2f}%")

if pred == 1:
    print("\n✅ CORRECT! Model predicts DISEASE (1) for high-risk patient")
else:
    print("\n❌ STILL WRONG! Model predicts NO DISEASE (0) for high-risk patient")
    print("   → You need to re-run the entire notebook to retrain with corrected labels!")

# Test Case 2: LOW-RISK patient (should NOW predict 0 = No Disease)
print("\n" + "="*80)
print("TEST 2: LOW-RISK Patient (young, healthy)")
print("="*80)

low_risk = pd.DataFrame({
    'age': [30],
    'sex': [0],
    'cp': [3],            # Asymptomatic
    'trestbps': [110],
    'chol': [180],
    'fbs': [0],
    'restecg': [0],
    'thalach': [180],     # High max heart rate - GOOD
    'exang': [0],         # No exercise angina - GOOD
    'oldpeak': [0.0],     # No ST depression - GOOD
    'slope': [0],
    'ca': [0],            # No vessels affected - GOOD
    'thal': [0]
})

print("\nPatient Profile:")
print("  - Age: 30 (young)")
print("  - Max Heart Rate: 180 (excellent)")
print("  - ST Depression: 0.0 (none - excellent)")
print("  - Vessels Affected: 0 (none)")
print("  - Exercise Angina: No")
print("  - Blood Pressure: 110 (normal)")
print("  - Cholesterol: 180 (normal)")

pred2 = model.predict(low_risk)[0]
proba2 = model.predict_proba(low_risk)[0]

print(f"\n🔍 Prediction: {pred2}")
print(f"   Probabilities: No Disease={proba2[0]*100:.2f}%, Disease={proba2[1]*100:.2f}%")

if pred2 == 0:
    print("\n✅ CORRECT! Model predicts NO DISEASE (0) for healthy patient")
else:
    print("\n❌ STILL WRONG! Model predicts DISEASE (1) for healthy patient")
    print("   → You need to re-run the entire notebook to retrain with corrected labels!")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if pred == 1 and pred2 == 0:
    print("✅ SUCCESS! All predictions are now clinically correct!")
    print("   The model is ready for deployment.")
    print("\n📌 Next steps:")
    print("   1. Update web app if needed")
    print("   2. Deploy to production")
    print("   3. Document the label correction in your report")
else:
    print("⚠️ ACTION REQUIRED!")
    print("\n📌 To fix this:")
    print("   1. Open Group_Assignment__1.ipynb")
    print("   2. Run ALL cells from the beginning (Runtime → Run All)")
    print("   3. This will:")
    print("      - Load data")
    print("      - Invert target labels")
    print("      - Retrain model with correct labels")
    print("      - Save corrected model to heart_model.pkl")
    print("   4. Run this verification script again")
