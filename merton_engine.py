
import yfinance as yf
import numpy as np
import pandas as pd

# target company 
ticker = "TSLA"
company = yf.Ticker(ticker)

# one year of past stock prices for the calculation of volatility 
print(f"Fetching market data for {ticker}...")
hist_data = company.history(period="1y")

market_cap = company.info.get('marketCap', 0)

# 'Total Liabilities Net Minority Interest' or fallback to standard metric
balance_sheet = company.balance_sheet
total_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] 

print("\n--- THE RAW MATERIALS ---")
print(f"Current Market Cap (Equity): ${market_cap:,.2f}")
print(f"Total Debt (Liabilities):    ${total_debt:,.2f}")
print(f"Trading Days Gathered:       {len(hist_data)} days")





# daily log returns
hist_data['Log_Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))

# SD of the returns just pulled 
daily_volatility = hist_data['Log_Return'].std()

#annulaizing the volatility
equity_volatility = daily_volatility * np.sqrt(252)

print("\n--- VOLATILITY ENGINE OUTPUT ---")
print(f"Daily Volatility:      {daily_volatility:.4%}")
print(f"Annualized Volatility: {equity_volatility:.4%}")


from scipy.stats import norm

# Black-Scholes parameters
V = market_cap + total_debt
D = total_debt
sigma = equity_volatility
r = 0.045
T = 1.0

# DD
# how many SD is firm's value away from debt line
d1 = (np.log(V / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
DD = d1 - sigma * np.sqrt(T)

# PD
# the probability (V < D)
PD = norm.cdf(-DD)

print("\n--- DEFAULT RISK METRICS ---")
print(f"Total Firm Value (V):    ${V:,.2f}")
print(f"Distance to Default:     {DD:.4f} standard deviations")
print(f"Probability of Default:  {PD:.4%} (over 1 year)")