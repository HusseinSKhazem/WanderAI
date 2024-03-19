from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

@app.route('/recommend_locations', methods=['POST'])
def recommend_locations_api():

    df = pd.read_csv('places.csv')  
    budget_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Budget'] = df['Budget'].map(budget_mapping)

    data = request.get_json(force=True)
    user_is_quiet = data['IsQuiet']
    user_has_beaches = data['HasBeaches']
    user_budget = data['Budget']

    user_preferences = {
        'IsQuiet': int(user_is_quiet),
        'HasBeaches': int(user_has_beaches),
        'Budget': int(user_budget)
    }


    filtered_df = df[(df[list(user_preferences.keys())] == pd.Series(user_preferences)).all(axis=1)]

    
    X_filtered = filtered_df.drop('Location', axis=1)
    y_filtered = pd.get_dummies(filtered_df['Location'])
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    models = {}
    for location in y_filtered.columns:
        model = RandomForestClassifier(n_estimators=1, random_state=42)
        model.fit(X_train, y_train[location])
        models[location] = model


    predictions = pd.DataFrame(index=X_test.index, columns=y_filtered.columns)
    for location in models:
        probs = models[location].predict_proba(X_test)
        predictions[location] = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]


    N = 3
    top_n_predictions = predictions.apply(lambda x: list(x.nlargest(N).index), axis=1)


    recommendations = top_n_predictions.to_dict()
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
