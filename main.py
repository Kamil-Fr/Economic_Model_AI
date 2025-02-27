import pandas as pd
from pycaret.regression import setup, compare_models, finalize_model
import pickle

# Load data
df = pd.read_csv('data/economic_data.csv')

# Debugowanie danych
print("Pierwsze 5 wierszy danych:")
print(df.head())
print("Kolumny w danych:")
print(df.columns)

# Sprawdzenie brakujących danych
print("Czy są brakujące dane?")
print(df.isnull().sum())

# Usuwanie brakujących danych
df = df.dropna()  # Możesz również użyć df.fillna(), aby uzupełnić brakujące wartości

# Usuń kolumnę 'Country' (jeśli jest kategoryczna)
df = df.drop(columns=['Country'])

# Upewnij się, że kolumna 'Economic Growth (%)' jest numeryczna
df['Economic Growth (%)'] = pd.to_numeric(df['Economic Growth (%)'], errors='coerce')

# Konfiguracja PyCaret
exp = setup(data=df, target='Economic Growth (%)', session_id=42, normalize=True)

# Trening modelu
best_model = compare_models()
final_model = finalize_model(best_model)

# Zapisz model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model zapisany w models/model.pkl")



