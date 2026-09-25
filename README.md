# Quantitative Credit Risk Modeling Portfolio

## Overview
This repository contains Python-based quantitative credit risk models designed to estimate Probability of Default (PD) and assess corporate creditworthiness. Built to mirror industry-standard commercial underwriting frameworks (such as Moody's CreditEdge and RiskCalc), this portfolio demonstrates the application of structural and fundamental credit modeling techniques.

The project bridges macroeconomic market data and fundamental accounting ratios to evaluate risk across both public and private middle-market exposures.

---

## Model 1: Merton Distance-to-Default (DD) Engine
**File:** `merton_engine.py`  
**Framework:** Structural Credit Modeling (Moody's CreditEdge equivalent)

This engine calculates the default risk of publicly traded companies by treating a firm's equity as a European call option on its underlying assets, utilizing the Black-Scholes-Merton framework.

*   **Market Data Ingestion:** Automates the retrieval of live equity pricing and balance sheet liabilities using `yfinance`.
*   **Volatility Engine:** Computes daily logarithmic returns and annualized equity volatility as a proxy for asset risk.
*   **Default Probability:** Utilizes `scipy.stats` to calculate the Distance-to-Default (DD) in standard deviations and outputs the 1-year Expected Default Frequency (EDF) / Probability of Default (PD).

## Model 2: Corporate Credit Rating Scorecard
**File:** `scorecard_engine.py`  
**Framework:** Fundamental Credit Scoring (Moody's RiskCalc equivalent)

A machine learning-driven underwriting scorecard that evaluates private and middle-market enterprises based on fundamental accounting ratios. 

*   **Financial Ratio Engineering:** Analyzes core credit metrics including Profitability (ROA), Leverage (Debt-to-Equity), and Liquidity (Current Ratio).
*   **Machine Learning Underwriting:** Trains a Logistic Regression model (`scikit-learn`) to determine the predictive weight of each financial ratio on historical default status, outputting a raw Probability of Default.
*   **Commercial Score Scaling:** Transforms raw ML probabilities into a standard 3-digit commercial credit score utilizing the industry-standard Points to Double the Odds (PDO) logarithmic scaling method.

---

## Technology Stack
*   **Language:** Python 3.x
*   **Data Processing:** `pandas`, `numpy`
*   **Statistical Math:** `scipy`
*   **Machine Learning:** `scikit-learn`
*   **Market Data API:** `yfinance`

---

## How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/GravityD9/merton_credit_model
   cd merton_credit_model

2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
source .venv/bin/activate    # Mac/Linux

3. Install dependencies
pip install yfinance numpy scipy pandas scikit-learn

4. Execute the models
python merton_engine.py
python scorecard_engine.py