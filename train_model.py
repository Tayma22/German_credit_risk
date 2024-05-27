import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load your dataset
df = pd.read_csv('german_data_credit_cat.csv')

# Select relevant features and target variable
features = ["Duration in month", "Credit history", "Purpose", "Credit amount", "Savings account/bonds",
            "Present employment since", "Personal status and sex", "Age in years",
            "Number of existing credits at this bank", "foreign worker"]
target = "Cost Matrix(Risk)"

# Filter dataset to include only relevant features and target
df_filtered = df[features + [target]]

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_filtered)

# Split the data into features (X) and target variable (y)
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
with open('credit_risk_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the columns
with open('model_columns.pkl', 'wb') as columns_file:
    pickle.dump(X.columns.tolist(), columns_file)
