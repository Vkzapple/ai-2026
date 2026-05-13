import tkinter as tk
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\evell\Downloads\wine\WineQT.csv')
df = df.drop(columns=['Id'])

for col in df.columns:
    if col != 'quality':
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

X = df.drop(columns=['quality', 'residual sugar', 'free sulfur dioxide', 'pH'])
y = df['quality']

split = int(0.8 * len(df))
X_train = X.iloc[:split].values
X_test  = X.iloc[split:].values
y_train = (y.iloc[:split].values >= 6).astype(int)
y_test  = (y.iloc[split:].values >= 6).astype(int)

def gini(y):
    if len(y) == 0:
        return 0
    p1 = np.sum(y == 1) / len(y)
    p0 = np.sum(y == 0) / len(y)
    return 1 - (p1**2 + p0**2)

def best_split(X, y):
    best_gain = -1
    best_col  = None
    best_val  = None
    gini_parent = gini(y)
    for col in range(X.shape[1]):
        for val in np.unique(X[:, col]):
            kiri  = y[X[:, col] <= val]
            kanan = y[X[:, col] >  val]
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

def build_tree(X, y, depth=0, max_depth=7):
    if len(np.unique(y)) == 1:
        node = Node()
        node.label = y[0]
        return node
    if depth == max_depth:
        node = Node()
        node.label = np.bincount(y).argmax()
        return node
    col, val, gain = best_split(X, y)
    if gain == 0:
        node = Node()
        node.label = np.bincount(y).argmax()
        return node
    kiri_mask  = X[:, col] <= val
    kanan_mask = X[:, col] >  val
    node = Node()
    node.col   = col
    node.val   = val
    node.kiri  = build_tree(X[kiri_mask],  y[kiri_mask],  depth+1, max_depth)
    node.kanan = build_tree(X[kanan_mask], y[kanan_mask], depth+1, max_depth)
    return node

def predict_satu(node, x):
    if node.label is not None:
        return node.label
    if x[node.col] <= node.val:
        return predict_satu(node.kiri, x)
    else:
        return predict_satu(node.kanan, x)

def predict(node, X):
    return np.array([predict_satu(node, x) for x in X])

tree   = build_tree(X_train, y_train)
y_pred = predict(tree, X_test)

akurasi   = np.sum(y_pred == y_test) / len(y_test) * 100
TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * (precision * recall) / (precision + recall)

root = tk.Tk()
root.title('Decision Tree - Wine Quality')
root.geometry('400x320')

tk.Label(root, text='Evaluasi Model Wine Quality',
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