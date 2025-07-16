#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error



# In[2]:


# Load the preprocessed data
df = pd.read_csv("./preprocessed_data.csv", index_col="Date")
df.index = pd.to_datetime(df.index)

print("Loaded preprocessed dataset:")
print(df.head())


# In[3]:


# Feature engineering: Create lag features
look_back = 5  # Number of previous days to consider
for i in range(1, look_back + 1):
    df[f'Close_Lag_{i}'] = df['Close'].shift(i)

df.dropna(inplace=True)


# In[4]:


# Prepare the data
X = df[[f'Close_Lag_{i}' for i in range(1, look_back + 1)]]
y = df['Close']


# In[5]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


# In[6]:


# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")


# In[7]:


# Save the trained model
import joblib
model_path = "./trained_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")


# In[8]:


# Make predictions
y_pred = model.predict(X_test)


# In[9]:


# Error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy = 100 - mape


# In[10]:


# Print metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Model Accuracy: {accuracy:.2f}%")


# In[11]:


# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label="Actual Price", color='blue')
plt.plot(y_pred, label="Predicted Price (Linear Regression)", linestyle='--', color='red')
plt.title("Actual vs Predicted Stock Prices - Linear Regression")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[12]:


# Predict the next day's price
last_5_days = df['Close'].iloc[-5:].values  # Get the last 5 closing prices
last_5_days_df = pd.DataFrame([last_5_days], columns=[f'Close_Lag_{i}' for i in range(1, 6)])

future_pred = model.predict(last_5_days_df)
print(f"Predicted next day's price: {future_pred[0]:.2f}")

