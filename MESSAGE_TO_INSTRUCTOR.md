# Message to Instructor: Dataset Labeling Concern

**Subject:** Question Regarding Heart Disease Dataset Target Variable Interpretation

---

Dear Professor,

I am writing to seek clarification on the heart disease dataset (`heart.csv`) used in our Group Assignment 1. During the model development and validation process, I discovered what appears to be **counterintuitive labeling** in the target variable that contradicts standard clinical understanding of cardiovascular disease.

## Discovery

While testing the trained Random Forest model (which achieved 100% accuracy on the test set), I noticed that predictions seemed inverted from clinical expectations:

- **High-risk patients** (elderly, multiple cardiovascular risk factors) were predicted as **target=0 ("No Disease")**
- **Low-risk patients** (young, no risk factors) were predicted as **target=1 ("Has Disease")**

## Data Analysis Findings

To investigate this anomaly, I performed a statistical comparison of patients in each target class:

### Patients with target=0 ("No Disease") - N=499
**Have WORSE health indicators:**
- Average Age: **56.6 years** (older)
- Max Heart Rate: **139.1 bpm** (lower - concerning for cardiac fitness)
- ST Depression (oldpeak): **1.60** (higher - indicates worse cardiac strain)
- Major Vessels Affected (ca): **1.16** (more vessels with narrowing)
- Exercise-Induced Angina (exang): **55%** prevalence (high)
- Resting Blood Pressure: **134.1 mmHg** (elevated)
- Cholesterol: **251.3 mg/dl** (elevated)

### Patients with target=1 ("Has Disease") - N=526
**Have BETTER health indicators:**
- Average Age: **52.4 years** (younger)
- Max Heart Rate: **158.6 bpm** (higher - better cardiac fitness)
- ST Depression (oldpeak): **0.57** (lower - less cardiac strain)
- Major Vessels Affected (ca): **0.37** (fewer vessels with narrowing)
- Exercise-Induced Angina (exang): **13%** prevalence (low)
- Resting Blood Pressure: **129.2 mmHg** (better)
- Cholesterol: **241.0 mg/dl** (better)

## Clinical Interpretation Issue

This pattern is **opposite** to standard cardiovascular disease risk profiles where we would expect:
- **Disease patients** to be older, have lower exercise capacity, more ST depression, more vessel involvement, and more exercise-induced symptoms
- **Healthy patients** to be younger, have higher max heart rate, less ST depression, fewer affected vessels, and fewer symptoms

## Test Case Example

When testing with a clearly high-risk patient profile:
```json
{
  "age": 70,
  "sex": 1,
  "cp": 3,
  "trestbps": 185,
  "chol": 390,
  "fbs": 1,
  "restecg": 2,
  "thalach": 95,      // Very low max heart rate
  "exang": 1,         // Exercise-induced angina present
  "oldpeak": 3.5,     // High ST depression
  "slope": 0,
  "ca": 3,            // All 3 major vessels affected
  "thal": 3
}
```

**Model Prediction:** target=0 ("No Disease") with 67.5% confidence

This prediction is **clinically correct** according to the dataset's labeling pattern, but **counterintuitive** from a medical perspective.

## Hypothesis

I believe this dataset may use **angiographic findings** rather than **clinical disease** as the target:

- **target=1** might indicate: *"Presence of significant arterial narrowing detected on coronary angiography in otherwise compensated patients"* (younger patients with good exercise tolerance despite narrowing)
- **target=0** might indicate: *"Poor exercise tolerance/symptomatic patients"* (possibly with heart failure, arrhythmias, or severe disease)

This would be consistent with certain cardiovascular research datasets where the diagnostic gold standard is angiographic narrowing (>50% stenosis), not clinical symptoms.

## Questions for Clarification

1. **What does the target variable actually represent?**
   - Is it presence of angiographic narrowing (>50% stenosis)?
   - Clinical diagnosis of coronary artery disease?
   - Presence of symptoms?
   - Something else?

2. **Should the labels be inverted?**
   - Is there a possibility the target variable was coded incorrectly in this version of the dataset?

3. **Dataset Source**
   - Which version of the heart disease dataset is this?
   - Cleveland? Hungarian? Switzerland? VA Long Beach?
   - Is there documentation available on the target variable definition?

## Impact on Assignment

Our model is technically **performing correctly** according to the data's actual labeling:
- 100% test accuracy
- Perfect confusion matrix
- Excellent ROC-AUC scores

However, the **clinical interpretation** of predictions requires understanding what the target truly represents. Our current model documentation states "Heart Disease: Presence of heart disease (0 = No, 1 = Yes)" which may be **misleading** if the labels have a different meaning.

## Request

Could you please clarify:
1. The correct interpretation of the target variable
2. Whether we should proceed with the current labeling or invert it
3. How we should document this in our final report

I want to ensure our model is not only statistically accurate but also clinically meaningful and properly documented.

Thank you for your guidance on this matter. I'm happy to discuss this further or provide additional analysis if needed.

---

**Attachments/Evidence:**
- Data analysis script: `analyze_data.py`
- Statistical comparison showing inverted patterns
- Test cases demonstrating the issue
- Full investigation report: `DATASET_LABEL_ISSUE.md`

Best regards,
[Your Name]
[Date: December 20, 2025]
