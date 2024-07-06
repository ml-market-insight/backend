from pymongo import MongoClient
from CODE.db_connection import get_db
from typing import List

db = get_db()
financial_collection = db['FinancialData']

def get_all_stocks():
    cursor = financial_collection.find({}, {"ticker": 1, "_id": 0})
    stocks = [doc['ticker'] for doc in cursor]
    return stocks

def fetch_stock_data(tickers: List[str]):
    cursor = financial_collection.find({"ticker": {"$in": tickers}})
    data = list(cursor)
    return data

def save_calculated_data(data):
    # Supposons que tu veuilles sauvegarder les données calculées dans une autre collection
    calculated_collection = db['CalculatedData']
    calculated_collection.insert_many(data)
    
# Simulated database service functions

def fetch_stock_data(tickers: List[str]):
    # Placeholder for fetching stock data for the given tickers
    # This should interact with a real database or external API to get stock data
    return {
        ticker: {"price": 100 + i, "data": "sample_data"}
        for i, ticker in enumerate(tickers)
    }