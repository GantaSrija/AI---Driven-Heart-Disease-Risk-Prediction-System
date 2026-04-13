import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Set style for premium look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Ensure images directory exists
os.makedirs('static/images', exist_ok=True)

# Load Dataset
print("Loading dataset...")
df = pd.read_csv('../heart_failure_clinical_records_dataset.csv')
TARGET = "DEATH_EVENT"

# 1. Correlation Heatmap
print("Generating Correlation Heatmap...")
plt.figure(figsize=(12, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Class Distribution (Before Balancing)
print("Generating Class Distribution...")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=TARGET, data=df, palette=['#10b981', '#ef4444'])
plt.title('Target Class Distribution (Imbalanced)', fontsize=14)
plt.xlabel('Death Event (0=No, 1=Yes)')
plt.ylabel('Count')
for i in ax.containers:
    ax.bar_label(i,)
plt.tight_layout()
plt.savefig('static/images/class_distribution.png', dpi=300)
plt.close()

# Prepare Data for Model Evaluation Plots
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Load constraints used in training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# Load Model
print("Loading model...")
model = joblib.load('model.pkl')

# 3. Feature Importance
print("Generating Feature Importance...")
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', dpi=300)
plt.close()

# 4. Confusion Matrix
print("Generating Confusion Matrix...")
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Survived', 'Deceased'],
            yticklabels=['Survived', 'Deceased'])
plt.title('Confusion Matrix (Test Set)', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('static/images/confusion_matrix.png', dpi=300)
plt.close()

# 5. ROC Curve
print("Generating ROC Curve...")
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/roc_curve.png', dpi=300)
plt.close()

print("All plots generated successfully.")
