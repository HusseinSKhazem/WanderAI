import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

if len(sys.argv) != 3:
    print("Usage: python script.py user_min_budget user_max_budget")
    sys.exit(1)

user_min_budget = float(sys.argv[1])
user_max_budget = float(sys.argv[2])

df = pd.read_csv(r'C:\Users\Dark\OneDrive\Desktop\AI\flights.csv')

X = df[['Country', 'Destination']]
y = df['BudgetMax']

column_transformer = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), ['Country', 'Destination'])
], remainder='passthrough')

X_encoded = column_transformer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_encoded)
df['PredictedBudgetMax'] = y_pred

suggested_destinations = df[(df['PredictedBudgetMax'] >= user_min_budget) & (df['PredictedBudgetMax'] <= user_max_budget)]

print(suggested_destinations[['Country', 'Destination']].to_csv(index=False))
