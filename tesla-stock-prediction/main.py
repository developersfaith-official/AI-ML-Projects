import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tdata = yf.download('TSLA', period='2y')
tdata['Target'] = tdata['Close'].shift(-1)
tdata = tdata[:-1] 
X = tdata[['Open', 'High', 'Low', 'Volume']]
y = tdata['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model train ho gaya!")
print("Accuracy:", model.score(X_test, y_test))
today = tdata.iloc[-1][['Open', 'High', 'Low', 'Volume']].values.reshape(1, -1)
pred = model.predict(today)
print(f"Kal ka predicted Close price: {pred[0]:.2f}")

y_pred = model.predict(X_test)
plt.figure(figsize=(14,7))
plt.plot(y_test.values, label="Actual Close", color="green", linewidth=2)
plt.plot(y_pred, label="Predicted Close", color="red", linestyle="--", linewidth=2)
plt.title("Tesla Stock Price Prediction using Linear Regression", fontsize=18, fontweight='bold')
plt.xlabel("Test Days", fontsize=14)
plt.ylabel("Price in USD ($)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tesla_prediction_graph.png", dpi=300, bbox_inches='tight')

plt.show()