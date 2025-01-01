from xgboost import XGBRegressor, cv, DMatrix
import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

class Model:

    def __init__(self):
        self.preprocessing = Preprocessing()
        self.df = self.preprocessing.preprocessing()
        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = self.preprocessing.lag_features()

    def cross_validate_model(self):
        # Define the model parameters
        params = {
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 300,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'subsample': 0.9,
            'objective': 'reg:squarederror'
        }

        # Convert the training data to DMatrix
        dtrain = DMatrix(self.X_train_scaled, label=self.y_train)

        # Perform cross-validation
        cv_results = cv(params, dtrain, num_boost_round=300, nfold=5, metrics='rmse', as_pandas=True, seed=42)

        print(f"Cross-Validation RMSE Scores:\n{cv_results}")
        print(f"Mean RMSE: {cv_results['test-rmse-mean'].iloc[-1]}")
        print(f"Standard Deviation of RMSE: {cv_results['test-rmse-std'].iloc[-1]}")

    def train_model(self):
        # Train the XGBoost Model
        model = XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=300, reg_alpha=0, reg_lambda=1, subsample=0.9)
        model.fit(self.X_train_scaled, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test_scaled)

        # Evaluation Metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        return y_pred, mae, rmse

    def plot_results(self, y_pred):
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(self.y_test.index, self.y_test, label="Actual", color="blue")
        plt.plot(self.y_test.index, y_pred, label="Predicted", color="red")
        plt.title("XGBoost 1-Day Ahead Forecast")
        plt.legend()
        st.pyplot(plt)

if __name__ == '__main__':
    model = Model()
    model.cross_validate_model()
    y_pred, mae, rmse = model.train_model()

    st.title("Bitcoin Price Forecasting")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    model.plot_results(y_pred)