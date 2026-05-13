import tkinter as tk
import numpy as np
import pandas as pd

# =====================
# LOAD & PREPROCESSING
# =====================
df = pd.read_csv(r'diabetes.csv')

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())

for col in df.columns:
    if col != 'Outcome':
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

x = df.drop(columns=['Outcome'])
y = df['Outcome']
split = int(0.8 * len(df))
x_train = x.iloc[:split].values
x_test  = x.iloc[split:].values
y_train = y.iloc[:split].values
y_test  = y.iloc[split:].values

# =====================
# DECISION TREE
# =====================
def gini(y):
    if len(y) == 0:
        return 0
    p1 = np.sum(y == 1) / len(y)
    p0 = np.sum(y == 0) / len(y)
    return 1 - (p1**2 + p0**2)

def best_split(x, y):
    best_gain = -1
    best_col  = None
    best_val  = None
    gini_parent = gini(y)
    for col in range(x.shape[1]):
        for val in np.unique(x[:, col]):
            kiri  = y[x[:, col] <= val]
            kanan = y[x[:, col] >  val]
            if len(kiri) == 0 or len(kanan) == 0:
                continue
            gini_child = (len(kiri)*gini(kiri) + len(kanan)*gini(kanan)) / len(y)
            gain = gini_parent - gini_child
            if gain > best_gain:
                best_gain = gain
                best_col  = col
                best_val  = val
    return best_col, best_val, best_gain

class Node:
    def __init__(self):
        self.col   = None
        self.val   = None
        self.kiri  = None
        self.kanan = None
        self.label = None

def build_tree(x, y, depth=0, max_depth=5):
    if len(np.unique(y)) == 1:
        node = Node()
        node.label = y[0]
        return node
    if depth == max_depth:
        node = Node()
        node.label = np.bincount(y).argmax()
        return node
    col, val, gain = best_split(x, y)
    if gain == 0:
        node = Node()
        node.label = np.bincount(y).argmax()
        return node
    kiri_mask  = x[:, col] <= val
    kanan_mask = x[:, col] >  val
    node = Node()
    node.col   = col
    node.val   = val
    node.kiri  = build_tree(x[kiri_mask],  y[kiri_mask],  depth+1, max_depth)
    node.kanan = build_tree(x[kanan_mask], y[kanan_mask], depth+1, max_depth)
    return node

def predict_satu(node, x):
    if node.label is not None:
        return node.label
    if x[node.col] <= node.val:
        return predict_satu(node.kiri, x)
    else:
        return predict_satu(node.kanan, x)

def predict(node, x):
    return np.array([predict_satu(node, xi) for xi in x])

# build & evaluasi
tree   = build_tree(x_train, y_train)
y_pred = predict(tree, x_test)

akurasi   = np.sum(y_pred == y_test) / len(y_test) * 100
TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * (precision * recall) / (precision + recall)

# =====================
# GUI TKINTER
# =====================
root = tk.Tk()
root.title('Evaluasi Decision Tree - Diabetes')
root.geometry('400x320')

tk.Label(root, text='Evaluasi Model Diabetes',
         font=('Arial', 14, 'bold')).pack(pady=10)

frame = tk.Frame(root)
frame.pack()

data = [
    ('Akurasi',   f'{akurasi:.2f}%'),
    ('Precision', f'{precision:.2f}'),
    ('Recall',    f'{recall:.2f}'),
    ('F1 Score',  f'{f1:.2f}'),
    ('TP', str(TP)),
    ('TN', str(TN)),
    ('FP', str(FP)),
    ('FN', str(FN)),
]

for i, (label, nilai) in enumerate(data):
    tk.Label(frame, text=label, width=15, anchor='w').grid(row=i, column=0, padx=10, pady=3)
    tk.Label(frame, text=nilai, width=15, anchor='w').grid(row=i, column=1, padx=10, pady=3)

root.mainloop()