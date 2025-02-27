import pandas as pd
from pycaret.regression import setup, compare_models, finalize_model
import pickle

# Wczytanie danych
df = pd.read_csv('data/economic_data.csv')
# Sprawdź dostępne kolumny w DataFrame
print("Kolumny w danych przed obróbką:")
print(df.columns)

# Usuwanie nadmiarowych spacji w nazwach kolumn
df.columns = df.columns.str.strip()

# Sprawdź kolumny po usunięciu nadmiarowych spacji
print("Kolumny w danych po obróbce:")
print(df.columns)

# Upewnij się, że wszystkie kolumny są w odpowiednim formacie
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['GDP (in billion USD)'] = pd.to_numeric(df['GDP (in billion USD)'], errors='coerce')
df['Inflation Rate (%)'] = pd.to_numeric(df['Inflation Rate (%)'], errors='coerce')
df['Unemployment Rate (%)'] = pd.to_numeric(df['Unemployment Rate (%)'], errors='coerce')
df['Economic Growth (%)'] = pd.to_numeric(df['Economic Growth (%)'], errors='coerce')

# Sprawdzenie, czy są jakiekolwiek brakujące dane
print("Czy są brakujące dane?")
print(df.isnull().sum())

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

# Usuwanie kolumny 'Country' (jeśli jest kategoryczna)
df = df.drop(columns=['Country'])

# Upewnij się, że kolumna 'Economic Growth (%)' jest numeryczna
df['Economic_Growth'] = pd.to_numeric(df['Economic_Growth'], errors='coerce')

# Konfiguracja PyCaret
exp = setup(data=df, target='Economic_Growth', session_id=42, normalize=True)

# Trening modelu
best_model = compare_models()
final_model = finalize_model(best_model)

# Zapisz model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model zapisany w models/model.pkl")



