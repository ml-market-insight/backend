#### commande pour exécuter l'application : python CODE/app.py
#### tester avec httpie : http GET http://127.0.0.1:5000/


from flask import Flask, jsonify, send_file
import pandas as pd
from bson import ObjectId  # Importez ObjectId si vous utilisez MongoDB
from db_connection import get_prediction_data , get_asset_full_name_data   # Assurez-vous que db_connection.py est correctement importé
from io import StringIO


app = Flask(__name__)

# Exemple de route pour retourner les données d'un modèle
@app.route('/fetchAllTickers')
def fetch_all_tickers():
    # Obtenir les données de prédiction à partir d'une fonction ou méthode (ex: get_prediction_data())
    prediction_data = get_prediction_data()  # Assurez-vous que cette fonction retourne un DataFrame pandas
    
    # Convertir les ObjectId en chaînes de caractères dans le DataFrame
    # Si nécessaire, ajustez cette logique en fonction de la structure de vos données
    if '_id' in prediction_data.columns:  # Assurez-vous que '_id' est la colonne contenant ObjectId
        prediction_data['_id'] = prediction_data['_id'].astype(str)

    asset_full_name_data = get_asset_full_name_data()
    merged_data = prediction_data.merge(asset_full_name_data, on='ticker', how='left')

    transformed_data = merged_data.rename(columns={
        'img_ticker': 'icon',
        'confidence_level': 'trust'
    })[['name', 'ticker', 'icon', 'trust']]

    # Convertir le DataFrame pandas en un dictionnaire
    data_dict = transformed_data.to_dict(orient='records') # merged_data.drop(["time_series_data", "img_prev"], axis=1).to_dict(orient='records')
    
    # Renvoyer les données JSON
    return jsonify(data_dict)



# Exemple de route pour accéder à des données d'un modèle
@app.route('/predict', methods=['POST'])
def predict():
    # Ajoutez ici la logique pour faire des prédictions si nécessaire
    return jsonify({"prediction": "Résultat de prédiction"})


@app.route('/download')
def download_file():
    file_path = 'MLMarketInsight_Report.pdf'
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

