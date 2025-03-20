import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler




columns_to_use = ['number of bedrooms', 'number of bathrooms', 'living area','number of floors','Price']
df = pd.read_csv('House Price India.csv', usecols=columns_to_use)
df.head()
# modified_df = df.drop(columns=['id','guestroom','basement','hotwaterheating','airconditioning','parking','furnishingstatus','prefarea'])
# 🔹 Define features and target variable
X = [['number of bedrooms', 'number of bathrooms', 'living area','number of floors']]  # Features
y = df["Price"]  # Target variable (house price)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")

# 🔹 Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"🔹 MAE: {mae}")
print(f"🔹 MSE: {mse}")
print(f"🔹 RMSE: {rmse}")
print(f"🔹 R² Score: {r2}")