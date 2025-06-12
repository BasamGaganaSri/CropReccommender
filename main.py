import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Debug message to confirm script is running
print("Script is running...")

# Sample crop dataset
data = {
    'Nitrogen': [90, 40, 45, 70, 60, 85, 50, 60],
    'Phosphorus': [42, 20, 30, 40, 45, 38, 35, 40],
    'Potassium': [43, 43, 40, 60, 50, 55, 48, 52],
    'Temperature': [22.4, 20.0, 25.5, 21.2, 28.0, 24.0, 23.5, 25.0],
    'Humidity': [82, 78, 75, 80, 70, 85, 76, 77],
    'pH': [6.5, 6.0, 6.2, 6.8, 6.7, 6.3, 6.4, 6.6],
    'Rainfall': [200, 150, 160, 180, 210, 190, 170, 175],
    'Crop': ['Rice', 'Maize', 'Barley', 'Tomato', 'Potato', 'Wheat', 'Sugarcane', 'Cotton']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode the Crop labels into numbers
label_encoder = LabelEncoder()
df['Crop_encoded'] = label_encoder.fit_transform(df['Crop'])

# Features and target
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y = df['Crop_encoded']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# Crop recommendation function
def recommend_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    pred_encoded = model.predict(input_features)
    crop_name = label_encoder.inverse_transform(pred_encoded)
    return crop_name[0]

# Example usage
if __name__ == "__main__":
    # Example input values
    N, P, K = 60, 40, 50
    temp = 23.0
    humidity = 75
    ph = 6.5
    rainfall = 180

    recommended = recommend_crop(N, P, K, temp, humidity, ph, rainfall)
    print(f"Recommended crop for the given conditions is: {recommended}")

