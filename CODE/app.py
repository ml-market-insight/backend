#### commande pour exécuter l'application : python CODE/app.py
#### tester avec httpie : http GET http://127.0.0.1:5000/


from flask import Flask, jsonify, send_file, request
from flask_cors import CORS, cross_origin
import pandas as pd
from bson import ObjectId  # Importez ObjectId si vous utilisez MongoDB
from db_connection import get_prediction_data , get_asset_full_name_data   # Assurez-vous que db_connection.py est correctement importé
from io import StringIO
import datetime
from simulation import simulation
import json
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},
     origins="*",
     methods=["GET", "POST", "UPDATE", "DELETE", "PUT"],
     allow_headers=["Content-Type", "Authorization",
                    "Access-Control-Allow-Credentials", "Access-Control-Allow-Origin",
                    "Access-Control-Allow-Headers", "x-access-token", "Origin", "Accept",
                    "X-Requested-With", "Access-Control-Request-Method",
                    "Access-Control-Request-Headers"])

@app.route('/fetchAllTickers')
@cross_origin()
def fetch_all_tickers():
    prediction_data = get_prediction_data()
    if '_id' in prediction_data.columns:  
        prediction_data['_id'] = prediction_data['_id'].astype(str)

    asset_full_name_data = get_asset_full_name_data()
    merged_data = prediction_data.merge(asset_full_name_data, on='ticker', how='left')

    transformed_data = merged_data.rename(columns={
        'img_ticker': 'icon',
        'confidence_level': 'trust'
    })[['name', 'ticker', 'icon', 'trust']]
    data_dict = transformed_data.to_dict(orient='records')
    
    return jsonify(data_dict)


@app.route('/UserSimulation', methods=['POST'])
def user_simulation():
    body = request.json
    tickers = body.get('tickers')
    # ON DOIT RECEVOIR UNE LISTE DE TICKER DE LA SORTE EX : ["AAPL", "MSFT", "NVDA", "TSLA"] pour ensuite faire nos simulation et on te retourne le json
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    result = simulation(tickers)
    portfolio = [{"ticker": ticker, "weight": weight} for ticker, weight in zip(result['assets'].split(', '), map(float, result['allocation'].split(', ')))]
    response = {
        "portfolio": portfolio,
        "rendement": result["rendement"]
    }
    
    return jsonify(response)



# Cette route sert à tester la route /UserSimulation, sur internet tu met http://127.0.0.1:5000//UserSimulation/test et ça t'affiches les résultats pour Mattéo
@app.route('/UserSimulation/test', methods=['GET'])
def test_user_simulation():
    url = 'http://127.0.0.1:5000/UserSimulation'
    tickers = ["GOOGL", "AMZN", "GOOG", "TSLA"]
    response = requests.post(url, json=tickers)  # Envoyer directement la liste des tickers en tant que JSON
    try:
        response.raise_for_status()  # Vérifier si la requête a réussi
        json_response = response.json()  # Essayer de décoder la réponse JSON
        return jsonify({
            "status_code": response.status_code,
            "json_response": json_response
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": str(e)
        }), 500



@app.route('/download')
def download_file():
    file_path = fr'../MLMarketInsight_Report.pdf' # matthieu tu met MLMarketInsight_Report.pdf en path
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

