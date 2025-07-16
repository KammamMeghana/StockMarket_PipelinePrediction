#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score


# In[116]:


# Load dataset (replace 'your_file.csv' with your dataset file path)
df = pd.read_csv("C:/Users/kumaraswamy kammam/Desktop/nasdaq/stocks/ARLP.csv")
print(df.size)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualize the closing price
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.title('Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Feature engineering
# Create lag features (e.g., Close price of the previous day)
df['Previous_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

# Prepare the data for linear regression
X = df[['Previous_Close']]  # Feature: Previous day's close price
y = df['Close']             # Target: Current day's close price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Price', marker='o')
plt.plot(y_pred, label='Predicted Price', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Predict future values (example)
last_close = df['Close'].iloc[-1]  # Get the last closing price
future_pred = model.predict([[last_close]])  # Predict the next day's price
print(f"Predicted next day's price: {future_pred[0]}")


# In[117]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Calculate R² Score (Accuracy Metric for Regression)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")


# In[ ]:





# In[ ]:




