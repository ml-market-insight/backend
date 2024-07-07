#### commande pour exécuter l'application : python BACKEND/CODE/app.py
#### tester avec httpie : http GET http://127.0.0.1:5000/



from flask import Flask, jsonify

app = Flask(__name__)

# Exemple de route pour tester l'API
@app.route('/')
def index():
    return jsonify({"message": "Bienvenue dans votre API Flask"})

# Exemple de route pour accéder à des données d'un modèle
@app.route('/predict', methods=['POST'])
def predict():
    # Ici vous pouvez ajouter la logique pour utiliser vos modèles ML dans ML_MODELS/
    # Exemple simplifié pour la démonstration
    return jsonify({"prediction": "Résultat de prédiction"})

if __name__ == '__main__':
    app.run(debug=True)
