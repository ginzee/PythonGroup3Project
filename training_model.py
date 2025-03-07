import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

# Ensure the processed CSV exists
csv_file = "mag7_processed_final3.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Error: {csv_file} not found. Ensure data processing is complete.")

# Load processed data
data = pd.read_csv(csv_file)
data['date'] = pd.to_datetime(data['date'])

# Define features and target
features_with_pe = ['close', 'p_e_ratio', 'sma_50']
y = data['next_day_direction']
X = data[features_with_pe]

# Balance the dataset using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter Tuning using Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 500]
}

grid_search = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Train Model using Best Parameters
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, "mag7_final_model.pkl")
print("Model saved to mag7_final_model.pkl")