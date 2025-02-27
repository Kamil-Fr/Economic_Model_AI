import pandas as pd
from pycaret.regression import setup, compare_models, finalize_model
import pickle

# Load data
df = pd.read_csv('data/economic_data.csv')

# Remove the 'Country' column (if it's categorical)
df = df.drop(columns=['Country'])

# Configure PyCaret
exp = setup(data=df, target='Economic Growth (%)', session_id=42, normalize=True)

# Train the model
best_model = compare_models()
final_model = finalize_model(best_model)

# Save the model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model saved in models/model.pkl")
