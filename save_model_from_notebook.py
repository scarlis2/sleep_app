# Add this to a new code cell in your Jupyter notebook after training the model

import joblib

# Save the trained Random Forest model
joblib.dump(rf, "rf_model.pkl")

# Save a sample input file using the same columns as model training data
X_test.head(20).to_csv("sample_input.csv", index=False)

print("Saved rf_model.pkl and sample_input.csv")
