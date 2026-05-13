import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

student_ids = [f'STU{str(i).zfill(4)}' for i in range(1, n+1)]
genders = np.random.choice(['Male', 'Female'], n)
ages = np.random.randint(15, 19, n)
study_hours = np.round(np.random.uniform(1, 10, n), 1)
attendance_pct = np.round(np.random.uniform(50, 100, n), 1)
sleep_hours = np.round(np.random.uniform(4, 10, n), 1)
final_score = np.round(np.random.uniform(40, 100, n), 1)
passed = (final_score >= 60).astype(int)

df = pd.DataFrame({
    'student_id': student_ids,
    'gender': genders,
    'age': ages,
    'study_hours': study_hours,
    'attendance_pct': attendance_pct,
    'sleep_hours': sleep_hours,
    'final_score': final_score,
    'passed': passed
})

# tambah missing values biar realistis
idx = np.random.choice(df.index, 50, replace=False)
df.loc[idx, 'attendance_pct'] = np.nan

df.to_csv('student_performance.csv', index=False)
print(df.head())
print(f'Total: {len(df)} data')