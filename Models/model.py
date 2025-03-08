import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv('creditcard.csv')

# Prepare features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model on scaled training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save the trained model and the scaler
joblib.dump(model, 'credit_card_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and scaler for testing
loaded_model = joblib.load('credit_card_fraud_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Transform the test set using the loaded scaler
X_test_scaled_loaded = loaded_scaler.transform(X_test)

# Make predictions on the test set
y_pred = loaded_model.predict(X_test_scaled_loaded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
