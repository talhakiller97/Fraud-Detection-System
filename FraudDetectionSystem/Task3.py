import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\Talha Saeed\PycharmProjects\FraudDetectionSystem\dataset\Auto_fraud.csv")

# Fix column name if necessary
fraud_column = "FraudFound_P"

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Ensure fraud column has only valid values
df = df[df[fraud_column].isin([0, 1])]

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=[fraud_column])
y = df[fraud_column]

# Check class distribution
fraud_count = y.value_counts()
print("Class distribution:", fraud_count)

# Handle class imbalance dynamically
if len(fraud_count) > 1 and fraud_count.min() > 5:
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, fraud_count.min() - 1))
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError:
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
else:
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

# Ensure at least two unique classes exist
if len(set(y_resampled)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
else:
    print("Warning: Only one class present after resampling. Model training might not be effective.")
    X_train, X_test, y_train, y_test = X_resampled, X_resampled, y_resampled, y_resampled

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Function to encode categorical values correctly
def encode_input(feature, value):
    if feature in label_encoders:
        if value in label_encoders[feature].classes_:
            return label_encoders[feature].transform([value])[0]
        else:
            print(f"Warning: Unknown category '{value}' for {feature}, using most frequent category.")
            return label_encoders[feature].transform([df[feature].mode()[0]])[0]
    return value  # Return as is for numerical values


# Command-line interface for testing
def test_fraud_detection():
    print("\nEnter details for fraud detection:")
    input_data = []
    for col in X.columns:
        value = input(f"{col} ({'Categorical' if col in label_encoders else 'Numeric'}): ")
        if col in label_encoders:
            input_data.append(encode_input(col, value))
        else:
            try:
                input_data.append(float(value))
            except ValueError:
                print(f"Invalid input for {col}, using default value 0.")
                input_data.append(0)

    input_array = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_array)
    print("Fraud Detected" if prediction[0] == 1 else "No Fraud Detected")


if __name__ == "__main__":
    test_fraud_detection()