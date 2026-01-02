import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
data_size = 500

temperature = np.random.normal(75, 10, data_size)
pressure = np.random.normal(50, 5, data_size)
vibration = np.random.normal(0.5, 0.1, data_size)
usage_hours = np.random.randint(100, 10000, data_size)
humidity = np.random.normal(40, 5, data_size)


failure = (temperature > 85) | (pressure > 60) | (vibration > 0.65) | (usage_hours > 8000)
failure = failure.astype(int)  # 1 = will fail soon, 0 = safe


df = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Vibration': vibration,
    'Usage_Hours': usage_hours,
    'Humidity': humidity,
    'Failure': failure
})

print("Sample Data:\n", df.head())


X = df[['Temperature', 'Pressure', 'Vibration', 'Usage_Hours', 'Humidity']]
y = df['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Predictive Maintenance')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


new_data = pd.DataFrame({
    'Temperature': [90],
    'Pressure': [55],
    'Vibration': [0.7],
    'Usage_Hours': [9500],
    'Humidity': [45]
})

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

if prediction[0] == 1:
    print("\n Warning: The machine is likely to fail soon. Schedule maintenance immediately.")
else:
    print("\n The machine is operating safely.")
