#Data gathering and Exploration

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import pickle
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', '{:.2f}'.format)

# Define the list of cryptocurrency tickers
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'BNB-USD']

# Download historical data for the specified tickers within a custom date range
start_date = '2014-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
crypto_df = pd.DataFrame()
for ticker in cryptocurrencies:
    data = yf.download(ticker, start=start_date, end=end_date)
    data['currency'] = ticker
    crypto_df = pd.concat([crypto_df,data])

# Drop any rows with missing data
crypto_df.dropna(inplace=True)

crypto_df.sort_values(by=['currency','Date'],ascending = True, inplace=True)

for lag in [1,7,14,21,30,60,120]:
    crypto_df[f'lag_{lag}'] = crypto_df.groupby('currency')['Close'].shift(lag)

crypto_df.index = pd.to_datetime(crypto_df.index)

# Extract Year, Month, and Day
crypto_df['year'] = crypto_df.index.year
crypto_df['month'] = crypto_df.index.month
crypto_df['day'] = crypto_df.index.day
crypto_df['quarter'] = crypto_df.index.quarter
crypto_df['weekday'] = crypto_df.index.weekday

crypto_df['is_weekend'] = (crypto_df.index.weekday >= 5).astype(int)
crypto_df['is_start_of_month'] = (crypto_df.index.day == 1).astype(int)
crypto_df['is_end_of_month'] = (crypto_df.index.is_month_end).astype(int)

for lag in [7,14,21,30,60,180]:
    crypto_df[f'rolling_{lag}'] = crypto_df.groupby('currency')['Close'].shift(lag).rolling(lag).mean()


# Create subplots for each currency
fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False)

# Define custom colors for each currency
colors = {'BTC-USD': 'blue', 'ETH-USD': 'green', 'ADA-USD': 'orange', 'XRP-USD': 'purple', 'BNB-USD': 'red'}

# Iterate through each currency and create a boxplot
for i, currency in enumerate(crypto_df['currency'].unique()):
    sns.boxplot(y='Close', data=crypto_df[crypto_df['currency'] == currency], ax=axes[i], color=colors[currency])
    axes[i].set_title(currency)
    axes[i].set_xlabel('Closing Price')

# Set common y-axis label
fig.text(0.04, 0.5, 'Closing Price', va='center', rotation='vertical')

plt.show()

unique_currencies = crypto_df['currency'].unique()

# for currency in unique_currencies:
#     plt.figure(figsize=(12, 6))
#     subset_df = crypto_df[crypto_df['currency'] == currency]
#     plt.plot(subset_df.index, subset_df['Close'], label=currency,color=colors[currency])
#     plt.title(f"{currency} - Close Price")
#     plt.xlabel("Date")
#     plt.ylabel("Close Price")
#     plt.legend()
#     plt.show()

crypto_df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder

# with open('encoder.pkl', 'wb') as file:
#     pickle.dump(le, file)

# Feature selection (assuming you want to predict 'Close' price)
X = crypto_df.drop(columns = ['Adj Close','Close']) # Features
y = crypto_df['Close']  # Target variable

le = LabelEncoder()
X['currency'] = le.fit_transform(X['currency'])

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 42)

Scaler = StandardScaler()
xtrain = Scaler.fit_transform(xtrain)
xtest = Scaler.transform(xtest)

# Initialize the models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(random_state=42)
svr_model = SVR()
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)

# Train and evaluate each models
models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
    "Support Vector Regression": svr_model,
    "Linear Regression": lr_model,
    "Decision Tree": dt_model
}

results_list = []

for name, model in models.items():
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    mse = mean_squared_error(ytest, y_pred)
    mae = mean_absolute_error(ytest, y_pred)
    r2 = r2_score(ytest, y_pred)

    results_list.append([name, mse, mae])

results_df = pd.DataFrame(results_list, columns=["Model", "MSE", "MAE"])

results_df = results_df.sort_values(by = 'MAE')
print(results_df)

X = Scaler.fit_transform(X)
lr_model = model.fit(X, y)



# #storing model and final csv
# with open('model_crypto.pkl', 'wb') as file:
#     pickle.dump(lr_model, file)

# with open('scaler_crypto.pkl', 'wb') as file:
#     pickle.dump(Scaler, file)

crypto_df.to_csv('data.csv')

# Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction")
    st.write("Predicting the next closing price of cryptocurrencies")

    st.sidebar.header("Select Cryptocurrency")
    currency_name = st.sidebar.selectbox("Pick Currency", ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'BNB-USD'])
    
    if st.button('Predict'):
        last_info = crypto_df[crypto_df['currency'] == currency_name].tail(1).drop(columns=[ 'Adj Close', 'Close'])
        last_info['currency'] = le.transform([currency_name])[0]
        last_info_scaled = Scaler.transform(last_info)

        predicted_price = lr_model.predict(last_info_scaled)

        st.write(f"Selected Cryptocurrency: {currency_name}")
        st.write(f"Predicted next closing price for {end_date}: ${predicted_price[0]:.2f}")

    st.sidebar.subheader("Visualization")
    chart_type = st.sidebar.radio("Select Chart Type", ["Candlestick", "Line"])
    st.sidebar.header("Select Timeframe")
    timeframe = st.sidebar.selectbox("Timeframe", ["7 days", "30 days", "60 days", "1 year"])

    days_map = {"7 days": 7, "30 days": 30, "60 days": 60, "1 year": 365}
    selected_days = days_map[timeframe]

    if currency_name:
        st.write(f"Showing data for the last {selected_days} days for {currency_name}")

        subset_df = crypto_df[crypto_df['currency'] == currency_name].tail(selected_days)

        if subset_df.empty:
            st.error(f"No data available for the last {selected_days} days for {currency_name}.")
        else:
            if chart_type == "Candlestick":
                fig = go.Figure(data=[go.Candlestick(x=subset_df.index,
                                                    open=subset_df['Open'],
                                                    high=subset_df['High'],
                                                    low=subset_df['Low'],
                                                    close=subset_df['Close'])])

                fig.update_layout(title=f"{currency_name} - Closing Prices for the last {selected_days} days",
                                xaxis_title="Date",
                                yaxis_title="Closing Price",
                                xaxis_rangeslider_visible=False,
                                template="plotly_white",
                                )
                st.plotly_chart(fig)
                
            elif chart_type == 'Line':
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(subset_df.index, subset_df['Close'], label=currency_name,color = 'cyan')
                ax.set_title(f"{currency_name} - Closing Prices for the last {selected_days} days")
                ax.set_xlabel("Date")
                ax.set_ylabel("Closing Price")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
            

if __name__ == "__main__":
    main()