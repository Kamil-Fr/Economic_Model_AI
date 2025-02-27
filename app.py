from flask import Flask, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Wczytywanie modelu raz przy uruchomieniu aplikacji
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Strona główna wyświetlająca index.html
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint API do pobierania danych i robienia prognoz
@app.route('/get_data', methods=['GET'])
def get_data():
    # Załaduj dane
    df = pd.read_csv('data/economic_data.csv')
    
    # Usuwanie nadmiarowych spacji w nazwach kolumn
    df.columns = df.columns.str.strip()

    # Debugowanie: wyświetlenie dostępnych kolumn
    print("Dostępne kolumny przed przekazaniem do modelu:")
    print(df.columns)

    # Upewnij się, że kolumny, które chcesz wybrać, istnieją
    required_columns = ['Year', 'GDP (in billion USD)', 'Inflation Rate (%)', 'Economic Growth (%)', 'Unemployment Rate (%)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return jsonify({'error': f"Brakujące kolumny: {missing_columns}"})
    
    # Przygotowanie danych do predykcji
    data_for_prediction = df[required_columns].values

    # Predykcja
    predictions = model.predict(data_for_prediction)

    # Przygotowanie danych do zwrócenia
    data = {
        'actual': df['Economic Growth (%)'].tolist(),
        'predicted': predictions.tolist()
    }
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
