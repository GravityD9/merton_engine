import pandas as pd
import numpy as np

# random data
np.random.seed(42)
n_companies = 1000

# fundamental financial ratios for 1,000 companies
# Profitability: Return on Assets (average 5%, std dev 8%)
roa = np.random.normal(0.05, 0.08, n_companies)

# Leverage: Debt to Equity (average 1.5x, std dev 1.0)
debt_equity = np.abs(np.random.normal(1.5, 1.0, n_companies))

# Liquidity: Current Ratio (average 1.2x, std dev 0.5)
current_ratio = np.abs(np.random.normal(1.2, 0.5, n_companies))

# default logic 
# High leverage, low ROA, and low current ratio increase default risk
hidden_risk_score = (debt_equity * 0.4) - (roa * 3.0) - (current_ratio * 0.5)

# conversion of score to probability and flagging the bottom ~20% as Defaults (1)
probabilities = 1 / (1 + np.exp(-hidden_risk_score))
default_status = (probabilities > 0.6).astype(int) 

# Pandas DataFrame
df = pd.DataFrame({
    'ROA': roa,
    'Debt_to_Equity': debt_equity,
    'Current_Ratio': current_ratio,
    'Default': default_status
})

print("--- CORPORATE FINANCIAL DATASET (TOP 5 ROWS) ---")
print(df.head())
print(f"\nTotal Companies Analyzed: {len(df)}")
print(f"Total Defaults in Dataset: {df['Default'].sum()}")



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#LOGISTIC REGRESSION 
# separating ratios from defualt status 
X = df[['ROA', 'Debt_to_Equity', 'Current_Ratio']]
y = df['Default']

# splitting 80% for training and 20% for testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize and train
clf = LogisticRegression()
clf.fit(X_train, y_train)

# extracting the coefficients
# figuring out the weightage laid by the ai on each financial ratio
print("\n--- UNDERWRITING SCORECARD WEIGHTS ---")
features = X.columns
for feature, coef in zip(features, clf.coef_[0]):
    print(f"{feature}: {coef:.4f}")
    
# testing the model on the unseen 20% of companies
y_pred_proba = clf.predict_proba(X_test)[:, 1] # Get the probability of default
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- MODEL PERFORMANCE ---")
print(f"ROC-AUC Score: {roc_auc:.4f} (1.0 is perfect, 0.5 is random guessing)")



# credit score scaling PDD method

# business parameters for the Scorecard
pdo = 20            # Points to Double the Odds
base_score = 600    # Anchor Score
base_odds = 50      # Odds of 50:1 at the Anchor Score

# scaling factor and offset as they're the standard log formulae used by Moody's, FICO and Fair Isaac
factor = pdo / np.log(2)
offset = base_score - (factor * np.log(base_odds))

# conversion of probabilities to Odds
odds = (1 - y_pred_proba) / y_pred_proba

# final credit score, rounding to the nearest whole number (indutry formats) 
credit_scores = offset + (factor * np.log(odds))
credit_scores = np.round(credit_scores).astype(int)

print("\n--- FINAL SCALED CREDIT SCORES (SAMPLE) ---")

# summary table for the first 10 tested companies
results_df = pd.DataFrame({
    'Predicted_PD': y_pred_proba,
    'Implied_Odds': odds,
    'Final_Credit_Score': credit_scores
}).head(10)

print(results_df.to_string(formatters={'Predicted_PD': '{:.2%}'.format, 'Implied_Odds': '{:.1f}:1'.format}))