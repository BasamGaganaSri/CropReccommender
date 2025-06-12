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
print(f"\nModel Accuracy: {acc*100:.2f}%")

# Crop recommendation function with neat output
def recommend_crop_with_proba(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    input_dict = {
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'pH': [pH],
        'Rainfall': [rainfall]
    }
    input_df = pd.DataFrame(input_dict)  # Ensure it's a DataFrame with column names
    pred_encoded = model.predict(input_df)
    proba = model.predict_proba(input_df)
    crop_name = label_encoder.inverse_transform(pred_encoded)
    
    # Neat output display
    print("\n--- Crop Prediction Summary ---")
    print(f"Predicted Crop: {crop_name[0]}")
    print("\nPrediction Probabilities for each crop:")
    for crop, probability in zip(label_encoder.classes_, proba[0]):
        print(f"{crop}: {probability*100:.2f}%")
    print("\n--------------------------------")

    return crop_name[0]

# Example usage
if __name__ == "__main__":
    # Input values for a sample recommendation
    N, P, K = 70, 40, 60  # Nitrogen, Phosphorus, Potassium
    temp = 21.2            # Temperature
    humidity = 80          # Humidity
    ph = 6.8               # pH
    rainfall = 180         # Rainfall

    recommended = recommend_crop_with_proba(N, P, K, temp, humidity, ph, rainfall)
    print(f"\nRecommended crop for the given conditions is: {recommended}")

