# 📈 Stock Market Prediction Pipeline using Azure ML & Linear Regression

This project implements an end-to-end stock price prediction pipeline using historical data. It fetches data from Azure Blob Datastore, preprocesses it, trains a Linear Regression model using lag features, evaluates its performance, and predicts the next day's closing price.

---

## 🔧 Technologies Used

- **Python**
- **Azure Machine Learning SDK**
- **Pandas, NumPy, Matplotlib**
- **Scikit-learn**
- **Joblib**

---

## 🗂️ Project Structure

├── data_preprocessing.py # Loads and preprocesses stock data from Azure Blob
├── model_training.py # Trains the Linear Regression model and evaluates it
├── pythonscript.py # Full pipeline with user input-based forecasting
├── preprocessed_data.csv # Output of preprocessing (optional)
├── trained_model.pkl # Saved scikit-learn model (optional)
├── requirements.txt # Dependencies list (you can auto-generate)
└── README.md # Project documentation



## ⚙️ Pipeline Workflow

### 1. **Data Preprocessing**
- Connects to Azure ML Workspace and Blob Datastore.
- Loads stock CSV files (e.g., `HEAR.csv`, `MSFT.csv`, etc.).
- Combines multiple stocks into a single DataFrame.
- Parses dates, sets datetime index, and selects latest 500 rows.
- Saves cleaned data to `preprocessed_data.csv`.

### 2. **Model Training**
- Loads cleaned data.
- Creates **lag-based features** (last 5 days’ closing prices).
- Splits into training and test sets.
- Trains a **Linear Regression** model.
- Calculates metrics: MSE, RMSE, MAE, MAPE, Accuracy.
- Saves the trained model as `trained_model.pkl`.

### 3. **Prediction**
- Accepts user input of 5 previous days' closing prices.
- Uses trained model to predict the 6th day's price.
- Includes plots of actual vs predicted stock prices.

---

## 📊 Sample Output

Mean Squared Error (MSE): 12.45
Root Mean Squared Error (RMSE): 3.52
Mean Absolute Error (MAE): 2.89
Model Accuracy: 94.35%


## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-pipeline.git
   cd stock-market-pipeline
Install dependencies:

pip install -r requirements.txt
Make sure your Azure Workspace config is set up properly.

Run scripts in order:

python data_preprocessing.py
python model_training.py
python pythonscript.py
✅ Future Improvements

Add forecasting for multiple days ahead.

Deploy as a web app or integrate with Azure Functions.

Add real-time data streaming via APIs or Kafka.

👩‍💻 Author
Meghana
Email: meghanakammam@gmail.com

