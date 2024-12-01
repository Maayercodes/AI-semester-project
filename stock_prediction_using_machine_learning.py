import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = 'AAPL_short_volume.csv'
data = pd.read_csv(file_path)

data = data.iloc[::-1].reset_index(drop=True)

def compute_technical_indicators(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

data = data[['Close']]
data = compute_technical_indicators(data)
data = data.dropna().reset_index(drop=True)

X = data[['SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'SVM': SVR()
}

params = {
    'LinearRegression': {}, 
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]},
    'XGBoost': {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]},
    'LightGBM': {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]},
    'SVM': {'C': [0.1, 1], 'gamma': [0.01, 0.1]}
}

results = {}
tscv = TimeSeriesSplit(n_splits=5)

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    if model_name == 'LinearRegression':
        best_model = model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
    else:
        grid_search = GridSearchCV(
            model, 
            params[model_name], 
            scoring='neg_mean_squared_error', 
            cv=tscv
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print(f"Best {model_name} Parameters: {grid_search.best_params_}")
    
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[model_name] = {
        'Model': best_model,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'MAPE': mape
    }

    print(f"{model_name} Results:")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - MAE: {mae:.2f}")
    print(f" - R^2: {r2:.2f}")
    print(f" - MAPE: {mape:.2f}%")
    print("------------------------")

best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['Model']
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", alpha=0.7)
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.title(f"Best Model: {best_model_name} - Actual vs. Predicted")
plt.xlabel("Samples")
plt.ylabel("Stock Prices")
plt.legend()
plt.show()

print(f"Best Model: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.2f}")
print(f"Evaluation: {results[best_model_name]}")
