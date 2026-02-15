import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv("dataset/mental_health.csv")

# Drop Timestamp (not useful for prediction)
data = data.drop("Timestamp", axis=1)

# Remove rows with missing values
data = data.dropna()

# Initialize Label Encoder
le = LabelEncoder()

# Encode all categorical columns
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Target column
target_column = "Do you have Depression?"

# Features and Target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("mental_model.pkl", "wb"))

print("\nModel saved successfully!")