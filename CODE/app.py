#### commande pour exécuter l'application : python BACKEND/CODE/app.py
#### tester avec httpie : http GET http://127.0.0.1:5000/

from db_connection import *

from flask import Flask, jsonify

app = Flask(__name__)

# Exemple de route pour tester l'API
@app.route('/fetchAllTickers')
def index():
    
    prediction = get_prediction_data()
    pred = pd.DataFrame()

    for i in range(len(prediction["ticker"])):
        asset = prediction.loc[i, "ticker"]
        time_series = pd.DataFrame(prediction.loc[i, "time_series_data"])
        pred[asset] = time_series["close"]
    

    return jsonify({pred})

# Exemple de route pour accéder à des données d'un modèle
@app.route('/predict', methods=['POST'])
def predict():
    # Ici vous pouvez ajouter la logique pour utiliser vos modèles ML dans ML_MODELS/
    # Exemple simplifié pour la démonstration
    return jsonify({"prediction": "Résultat de prédiction"})

if __name__ == '__main__':
    app.run(debug=True)
