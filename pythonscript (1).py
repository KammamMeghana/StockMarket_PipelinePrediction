from datetime import datetime, date
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from azureml.core import Workspace, Dataset, Datastore

# Set up workspace
print("Date of pipeline ML code run is:", date.today())
ws = Workspace.from_config()

# Access datastore
datastore = Datastore.get(ws, 'blobconnection')

# List of CSV files in the datastore
csv_files = [
    "SSRM.csv",
    "ARLP.csv",
    "HEAR.csv",
    "MSFT.csv",
    "PEP.csv",
    "VUZI.csv",
]

# Combine all CSV files into one DataFrame
dataframes = []
for file_name in csv_files:
    dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, file_name)])
    df = dataset.to_pandas_dataframe()
    dataframes.append(df)

# Combine all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df.set_index('Date', inplace=True)

# Display the first few rows
print("Combined DataFrame:")
print(combined_df.head())

# Plot closing price history
plt.figure(figsize=(14, 7))
plt.plot(combined_df['Close'], label='Closing Price')
plt.title('Stock Closing Price History')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Feature engineering: Create lag features
look_back = 5  # Number of previous days to consider
for i in range(1, look_back + 1):
    combined_df[f'Close_Lag_{i}'] = combined_df['Close'].shift(i)

combined_df.dropna(inplace=True)

# Prepare data for modeling
X = combined_df[[f'Close_Lag_{i}' for i in range(1, look_back + 1)]]
y = combined_df['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label="Actual Price", color='blue')
plt.plot(y_pred, label="Predicted Price", linestyle='--', color='red')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# User input for predicting the 6th day's price from 5 days' data
while True:
    try:
        print("\nEnter closing prices for the last 5 days:")
        last_5_days = []
        for i in range(1, 6):
            price = float(input(f"Day {i} closing price: "))
            last_5_days.append(price)

        # Convert to DataFrame with appropriate column names
        last_5_days_df = pd.DataFrame([last_5_days], columns=[f'Close_Lag_{i}' for i in range(1, 6)])

        # Predict the 6th day's price using the trained model
        prediction = model.predict(last_5_days_df)
        print(f"Predicted closing price for the 6th day: {prediction[0]:.2f}")
    except Exception as e:
        print(f"Error: {e}")

    cont = input("Do you want to predict next day's price? (yes/no): ").strip().lower()
    if cont != 'yes':
        break
