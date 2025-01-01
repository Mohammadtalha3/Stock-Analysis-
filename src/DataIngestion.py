import yfinance as yf
import pandas as pd
import csv# Example for BTC




def data_Ingestion():
    data = yf.download('BTC-USD', start='2014-09-17', end='2024-12-29')
    data.to_csv('BTC.csv')
    return data


dt= data_Ingestion()
print(dt.head(10))



    










