# Tesla Stock Price Prediction using Linear Regression

> Ek beginner-friendly project jo sirf **Linear Regression** se Tesla ka **next day Close price** predict karta hai  
> Accuracy ~98% | 100% working code | Step-by-step samjhaaya gaya

🌐 **Read in:** [English](#english) | [Roman Urdu](#roman-urdu) 
---



---
### English
## What This Project Does
- Downloads **2 years of Tesla stock data** from Yahoo Finance automatically
- Creates a `Target` column → tomorrow's Close price
- Trains a Linear Regression model
- Predicts **tomorrow's Close price live**
- Plots beautiful **Actual vs Predicted** graph
---

## Requirements (Run once)
```bash
pip install yfinance scikit-learn matplotlib pandas numpy
```
How to Run (5 Minutes)

1. Download or clone this repository
2. Open terminal in the folder
3. Run:
```bash
pip install yfinance scikit-learn matplotlib pandas numpy
python main.py
```
## Import required libraries
``` python
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```
## Downloads last 2 years of Tesla stock data
```python
tdata = yf.download('TSLA', period='2y')
```
## Creates Target = Tomorrow's Close price
## Removes last row (its target was NaN)
```python
tdata['Target'] = tdata['Close'].shift(-1)
tdata = tdata[:-1]
```
## X = what we give to model, y = what we want to predict
```python
X = tdata[['Open', 'High', 'Low', 'Volume']]   # Today's data (input/features)
y = tdata['Target']                            # Tomorrow's Close (output/label)
```
## Splits data: 80% training, 20% testing
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Trains the model and shows accuracy (~98%)
```python
model = LinearRegression()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```
# Predict tomorrow's price
```python
today = tdata.iloc[-1][['Open','High','Low','Volume']].values.reshape(1,-1)
pred = model.predict(today)
print(f"Tomorrow's predicted Close price: {pred[0]:.2f}")
```
# Plot Actual vs Predicted
```python
# Plot Actual vs Predicted
y_pred = model.predict(X_test)
plt.figure(figsize=(14,7))
plt.plot(y_test.values, label="Actual Price", color="green", linewidth=2)
plt.plot(y_pred, label="Predicted Price", color="red", linestyle="--", linewidth=2)
plt.title("Tesla Stock Price Prediction - Linear Regression", fontsize=18, fontweight='bold')
plt.xlabel("Test Days", fontsize=14)
plt.ylabel("Price in USD ($)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tesla_prediction_graph.png", dpi=300, bbox_inches='tight')
plt.show()
```
---
### Roman Urdu 
### Project Kya Karta Hai?
- Yahoo Finance se Tesla ka 2 saal ka data lata hai
- `Target` column banata hai → kal ka Close price
- Linear Regression model train karta hai
- Live next day prediction deta hai
- Actual vs Predicted ka sundar graph banata hai
### Requirements (pehle yeh install karo)
```bash
pip install yfinance scikit-learn matplotlib pandas numpy
```
## Kaise Chalayein? (Step-by-Step)
1. File download karo → main.py
2. Terminal mein jao → python main.py
3. Output dekho:
 - Model accuracy
 - Kal ka predicted price
 - Graph khud khul jayega

## import yfinance python library jo yahoo finance sa financial data access krny ma help krta hai.
``` python
import yfinance as yf
```
## yahoo sa 2 saal ka data lata hain
```python
tdata = yf.download('TSLA', period='2y')# ← TSLA ak ticket ha jo tasla k aset ko represent krta ha or period=2  2-saal ka data nikly ga.
```

## Kal ka Close Price
``` python
tdata['Target'] = tdata['Close'].shift(-1) 
#tdata['Target'] ← ak new column bny ga jis k name target ho ga or is ma tdata['Close'] column ka data a jay ga.
# .shift[-1] key fuction ha jo ak row upper close k column ma data shift kr rha ha.
```
## Last row hatain
``` python
tdata = tdata[:-1] # last row hatao (jiska target nahi hai)
```
## Data assign two variable
``` python
X = tdata[['Open', 'High', 'Low', 'Volume']] # Aaj ka data (input)
y = tdata['Target']                          # Kal ka Close (jo predict karna hai)
```
## Model ko bataya ke "yeh 4 cheezen dekh kar kal ka price bata"
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Model train (Model ne 2 saal ka pattern seekh liya)
``` python
model = LinearRegression()
model.fit(X_train, y_train)
```
## Kal ka prediction (Aaj ke last din ke data se kal ka price predict kiya)
``` python
today = tdata.iloc[-1][['Open','High','Low','Volume']].values.reshape(1,-1)
pred = model.predict(today)
print(f"Kal ka predicted Close price: {pred[0]:.2f}")
```
## Actual vs Predicted ka sundar graph + PNG save kiya
```python 
# Graph banaya
plt.plot(y_test.values, label="Actual", color="green")
plt.plot(model.predict(X_test), label="Predicted", color="red", linestyle="--")
plt.title("Tesla Stock Price Prediction")
plt.legend()
plt.savefig("tesla_prediction_graph.png", dpi=300)
plt.show()
```