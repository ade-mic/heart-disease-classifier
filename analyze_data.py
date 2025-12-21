import pandas as pd

df = pd.read_csv('heart.csv')

print("="*80)
print("ANALYZING PATIENTS WITH target=0 (No Disease)")
print("="*80)
no_disease = df[df['target'] == 0]
print(f'Count: {len(no_disease)}')
print(f'\nAverage values for NO DISEASE patients:')
print(f'  Age: {no_disease["age"].mean():.1f}')
print(f'  Max Heart Rate: {no_disease["thalach"].mean():.1f}')
print(f'  Cholesterol: {no_disease["chol"].mean():.1f}')
print(f'  BP: {no_disease["trestbps"].mean():.1f}')
print(f'  ST Depression (oldpeak): {no_disease["oldpeak"].mean():.2f}')
print(f'  Vessels (ca): {no_disease["ca"].mean():.2f}')
print(f'  Exercise Angina (exang): {no_disease["exang"].mean():.2f}')

print(f'\n{"="*80}')
print('ANALYZING PATIENTS WITH target=1 (Has Disease)')
print("="*80)
disease = df[df['target'] == 1]
print(f'Count: {len(disease)}')
print(f'\nAverage values for HAS DISEASE patients:')
print(f'  Age: {disease["age"].mean():.1f}')
print(f'  Max Heart Rate: {disease["thalach"].mean():.1f}')
print(f'  Cholesterol: {disease["chol"].mean():.1f}')
print(f'  BP: {disease["trestbps"].mean():.1f}')
print(f'  ST Depression (oldpeak): {disease["oldpeak"].mean():.2f}')
print(f'  Vessels (ca): {disease["ca"].mean():.2f}')
print(f'  Exercise Angina (exang): {disease["exang"].mean():.2f}')

print(f'\n{"="*80}')
print('KEY INSIGHT:')
print("="*80)
print('Patients with HIGHER heart rate and LOWER oldpeak/ca/exang')
print('are labeled as having disease (target=1).')
print('\nThis suggests target=1 might mean "presence of narrowing"')
print('detected on angiogram, NOT clinical disease symptoms!')
