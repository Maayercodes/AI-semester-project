import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_and_preprocess_data(file_path):
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

    data = compute_technical_indicators(data)
    data = data.dropna().reset_index(drop=True)
    
    return data

def prepare_features_and_target(data):
    X = data[['SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal']]

    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.2f}")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.2f}")

    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2:.2f}")

    return rmse, mae, r2

def cross_validate_model(model, X, y):
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation RMSE scores: {-cross_val_scores}")
    print(f"Mean RMSE: {-cross_val_scores.mean():.2f}")

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def visualize_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs. Predicted Stock Prices (Random Forest)")
    plt.xlabel("Samples")
    plt.ylabel("Stock Prices")
    plt.legend()
    plt.show()

def main():
    file_path = 'AAPL_short_volume.csv'
    data = load_and_preprocess_data(file_path)

    X_train, X_test, y_train, y_test = prepare_features_and_target(data)

    rf_model = train_random_forest(X_train, y_train)

    rmse, mae, r2 = evaluate_model(rf_model, X_test, y_test)

    cross_validate_model(rf_model, X_train, y_train)

    best_rf_model = tune_hyperparameters(X_train, y_train)

    save_model(best_rf_model, 'stock_prediction_model.pkl')

    visualize_predictions(y_test, rf_model.predict(X_test))

if __name__ == "__main__":
    main()
