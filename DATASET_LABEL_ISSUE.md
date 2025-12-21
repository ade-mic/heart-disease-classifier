# ⚠️ IMPORTANT: Dataset Label Investigation

## Problem Discovery

The heart disease prediction model is making predictions that seem counterintuitive:
- High-risk patients (elderly, multiple risk factors) → Predicted as "No Disease" (0)
- Healthy patients (young, no risk factors) → Predicted as "Disease" (1)

## Root Cause Analysis

After analyzing the training data (`heart.csv`), we discovered the labels don't match typical clinical expectations:

### Target = 0 ("No Disease") Patients
**Have WORSE health indicators:**
- Average Age: **56.6 years** (older)
- Max Heart Rate: **139.1 bpm** (lower - concerning)
- ST Depression: **1.60** (higher - worse)
- Vessels Affected: **1.16** (more vessels)
- Exercise Angina: **55%** have it (higher)
- Blood Pressure: **134.1 mmHg** (higher)

### Target = 1 ("Has Disease") Patients  
**Have BETTER health indicators:**
- Average Age: **52.4 years** (younger)
- Max Heart Rate: **158.6 bpm** (higher - better)
- ST Depression: **0.57** (lower - better)
- Vessels Affected: **0.37** (fewer vessels)
- Exercise Angina: **13%** have it (lower)
- Blood Pressure: **129.2 mmHg** (lower/better)

## Hypothesis

The dataset might use **angiographic narrowing** as the target rather than clinical disease:

- **Target = 1**: Patients with **detected arterial narrowing on angiogram** but who can still exercise well (compensated, asymptomatic)
- **Target = 0**: Patients with **poor exercise tolerance** and symptoms (symptomatic heart disease, or other conditions)

## Example: Your Test Patient

```json
{
  "age": 70,
  "sex": 1,
  "cp": 3,  
  "trestbps": 185,
  "chol": 390,
  "fbs": 1,
  "restecg": 2,
  "thalach": 95,      ← Very low (matches target=0 pattern)
  "exang": 1,         ← Has exercise angina (matches target=0)
  "oldpeak": 3.5,     ← Very high (matches target=0)
  "slope": 0,
  "ca": 3,            ← Many vessels (matches target=0)
  "thal": 3
}
```

**Model Prediction: 0 (67.5% confidence)**

This patient's profile matches the **target=0** pattern from the training data:
- Low max heart rate (95 vs avg 139 for target=0)
- High ST depression (3.5 vs avg 1.60 for target=0)
- Many vessels affected (3 vs avg 1.16 for target=0)
- Has exercise angina (like 55% of target=0 patients)

## Solutions

### Option 1: Verify Dataset Source
Check the original dataset documentation to understand what "target" actually represents.

Common heart disease datasets:
- **Cleveland Heart Disease Dataset**: target=0-4 (severity levels), often binarized
- Some versions have inverted labels

### Option 2: Re-label the Dataset
If confirmed that labels are inverted, flip them:

```python
df['target'] = 1 - df['target']  # Invert labels
```

### Option 3: Update Model Interpretation
Keep the model as-is but update documentation:
- Rename target=1 to "Detected Narrowing (Compensated)"
- Rename target=0 to "Poor Exercise Tolerance / Symptomatic"

### Option 4: Get Different Dataset
Use a properly labeled clinical dataset where:
- target=1 = clinically diagnosed heart disease
- target=0 = healthy/no disease

## Recommended Action

1. **Check the notebook** - verify if the original Google Colab dataset was different
2. **Review dataset source** - confirm what "target" means in your `heart.csv`
3. **Update labels if needed** - flip target values if confirmed inverted
4. **Retrain model** - regenerate `heart_model.pkl` after fixing labels

## Files to Check
- `heart.csv` - verify this is the correct dataset
- `Group_Assignment__1.ipynb` - check original data source (was from Google Drive)
- Original dataset documentation

---

**Status**: Model is working correctly according to how it was trained. The issue is with dataset labeling, not the model itself.
