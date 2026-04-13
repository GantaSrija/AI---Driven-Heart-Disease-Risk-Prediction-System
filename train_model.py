import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv('../heart_failure_clinical_records_dataset.csv')

# Features and Target
TARGET = "DEATH_EVENT"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Imbalance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# Train Model (Best parameters from notebook analysis: max_depth=10, n_estimators=100)
# We use the defaults or close to defaults as starting point, but let's stick to what was found or a robust default.
# The notebook analysis output showed Random Forest performing well.
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Save Model and Scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully.")
