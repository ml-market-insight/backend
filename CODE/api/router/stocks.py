from fastapi import APIRouter, HTTPException
from typing import List
from CODE.ML_Models.markowitz import *
from CODE.api.models.stock import StockSelection
from CODE.api.services.db_service import (
    get_all_stocks,
    fetch_stock_data,
    save_calculated_data
)



router = APIRouter()

@router.get("/stocks")
async def get_stocks():
    try:
        stocks = get_all_stocks()
        return {"available_stocks": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/selectedstocks")
async def calculate_portfolio(stock_selection: StockSelection):
    try:
        tickers = stock_selection.tickers
        
        # Vérifiez que les tickers sont bien une liste de chaînes non vides
        if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers) or len(tickers) == 0:
            raise ValueError("Les tickers sélectionnés ne sont pas valides")

        # Appel à la fonction pour récupérer les données des stocks en fonction des tickers
        data = fetch_stock_data(tickers)

        # Vérifiez que des données ont été récupérées
        if not data:
            raise ValueError("Aucune donnée trouvée pour les tickers spécifiés")

        # Vous pouvez maintenant continuer avec le traitement des données récupérées
        # Par exemple :
        # - Calculer les rendements
        # - Générer et optimiser le portefeuille
        # - Sauvegarder les données calculées, etc.

        return {"status": "success", "data": data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/portfolio/results")
async def get_portfolio_results():
    # Supposons que vous avez déjà les résultats disponibles dans un format approprié
    results = {
        "portfolio_1": {
            "return": 0.08,
            "volatility": 0.12,
            "sharpe_ratio": 0.67
        },
        "portfolio_2": {
            "return": 0.07,
            "volatility": 0.11,
            "sharpe_ratio": 0.63
        }
        # Ajoutez d'autres résultats selon vos besoins
    }
    return {"status": "success", "data": results}