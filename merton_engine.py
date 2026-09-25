import yfinance as yf
import numpy as np
import pandas as pd

ticker = "TSLA"
company = yf.Ticker(ticker)

print(f"Fetching market data for {ticker}...")
hist_data = company.history(period="1y")

market_cap = company.info.get('marketCap', 0)

balance_sheet = company.balance_sheet
total_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] 

print("\n--- THE RAW MATERIALS ---")
print(f"Current Market Cap (Equity): ${market_cap:,.2f}")
print(f"Total Debt (Liabilities):    ${total_debt:,.2f}")
print(f"Trading Days Gathered:       {len(hist_data)} days")