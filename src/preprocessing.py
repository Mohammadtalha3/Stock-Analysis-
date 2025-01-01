import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
# from DataIngestion import data_Ingestion
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Preprocessing:
    def __init__(self):
        self.df = pd.read_csv('D:\\Stock_Anlysis\\btc_data.csv', parse_dates=['Date'], dayfirst=True)
        self.index = self.df.set_index('Date', inplace=True)
    
    def preprocessing(self):
        self.df['Price_Range'] = self.df['High'] - self.df['Low']
        self.df['Net_Change'] = self.df['Close'] - self.df['Open']
        self.df['Normalized_Range'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['Day_of_Week'] = self.df.index.dayofweek
        self.df['Month'] = self.df.index.month
        self.df['Year'] = self.df.index.year
        return self.df
    
    def lag_features(self):

        for lag in range(1, 8):  # Create lags for the past 7 days
            self.df[f'Adj_Close_Lag{lag}'] = self.df['Adj Close'].shift(lag)
        
        self.df.dropna(inplace=True)
        self.df['MA_7'] = self.df['Adj Close'].rolling(window=7).mean()

        self.df['MA_14'] = self.df['Adj Close'].rolling(window=14).mean()

        self.df['Target'] = self.df['Adj Close'].shift(-1)  # Shift -1 for 1-day ahead forecasting
        self.df = self.df.dropna() 

        columns_to_keep = ['Price_Range', 'Net_Change', 'Normalized_Range', 'Day_of_Week', 'Adj_Close_Lag1','MA_7']

        X = self.df[columns_to_keep]
        y = self.df['Target']


        train_size = int(0.8 * len(self.df))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        print('This is the lenght of the x_train after split ',len(X_train))

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print('This is the lenght of the x_train_scaled  after split ',len(X_train))

        return X_train_scaled, X_test_scaled,y_train, y_test
    


# dt= Preprocessing()
# cl=dt.preprocessing()
# X_train_scaled, X_test_scaled,X_train, y_train,X_test, y_test=dt.lag_features()


# # Parameter grid
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8, 0.9],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [1, 5, 10]
# }

# best_params = None
# best_score = float('inf')

# for max_depth in param_grid['max_depth']:
#     for learning_rate in param_grid['learning_rate']:
#         for n_estimators in param_grid['n_estimators']:
#             for subsample in param_grid['subsample']:
#                 for colsample_bytree in param_grid['colsample_bytree']:
#                     for reg_alpha in param_grid['reg_alpha']:
#                         for reg_lambda in param_grid['reg_lambda']:
#                             model = XGBRegressor(
#                                 max_depth=max_depth,
#                                 learning_rate=learning_rate,
#                                 n_estimators=n_estimators,
#                                 subsample=subsample,
#                                 colsample_bytree=colsample_bytree,
#                                 reg_alpha=reg_alpha,
#                                 reg_lambda=reg_lambda
#                             )
#                             model.fit(X_train_scaled, y_train)
#                             y_pred = model.predict(X_test_scaled)
#                             mse = mean_squared_error(y_test, y_pred)
#                             if mse < best_score:
#                                 best_score = mse
#                                 best_params = {
#                                     'max_depth': max_depth,
#                                     'learning_rate': learning_rate,
#                                     'n_estimators': n_estimators,
#                                     'subsample': subsample,
#                                     'colsample_bytree': colsample_bytree,
#                                     'reg_alpha': reg_alpha,
#                                     'reg_lambda': reg_lambda
#                                 }

#     print("Best Parameters:", best_params)
#     print("Best MSE:", best_score)




