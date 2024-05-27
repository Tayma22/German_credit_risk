from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__) 

# Load the trained model
with open('credit_risk_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the model columns
with open('model_columns.pkl', 'rb') as columns_file:
    model_columns = pickle.load(columns_file)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/webapp')
def fill():
    return render_template('webapp.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        duration = float(request.form["duration"])
        credit_history = request.form["credit_history"]
        purpose = request.form["purpose"]
        credit_amount = float(request.form["credit_amount"])
        savings_account = request.form["savings_account"]
        employment_since = request.form["employment_since"]
        personal_status = request.form["personal_status"]
        age = float(request.form["age"])
        existing_credits = float(request.form["existing_credits"])
        foreign_worker = request.form["foreign_worker"]
        
        # Create a DataFrame with the form data
        data = {
            "Duration in month": [duration],
            "Credit history": [credit_history],
            "Purpose": [purpose],
            "Credit amount": [credit_amount],
            "Savings account/bonds": [savings_account],
            "Present employment since": [employment_since],
            "Personal status and sex": [personal_status],
            "Age in years": [age],
            "Number of existing credits at this bank": [existing_credits],
            "foreign worker": [foreign_worker]
        }
        input_data = pd.DataFrame(data)
        
        # One-hot encode categorical variables
        input_data_encoded = pd.get_dummies(input_data)
        
        # Ensure all expected columns are present, filling missing ones with 0
        input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_data_encoded)
        
        # Determine the result
        result = "non-risky" if prediction[0] == 1 else "risky"
        
        return render_template('result.html', result=f'Your credit application is {result}!')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
