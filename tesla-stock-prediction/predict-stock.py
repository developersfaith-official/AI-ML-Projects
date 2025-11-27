import yfinance as yf
data = yf.download('TSLA', period='2y')
data.to_csv('download-tesla.csv')
data['Target'] = data['Close'].shift(-1)
# data = data[:-1]
data = data.dropna()
print(data.tail())