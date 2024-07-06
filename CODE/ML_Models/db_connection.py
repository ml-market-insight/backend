from pymongo.server_api import ServerApi
from packages_import import *
import pandas as pd

client = MongoClient('mongodb+srv://matthieuvichet:aO0mNOu20DVJpKHz@mlmarketinsights.vgtptz0.mongodb.net/')
db = client['MLMarketInsights']
financial_collection = db['FinancialData']

def insert_financial_data(df):
    time_series_data = df.drop(columns=['ticker', 'exchange']).to_dict('records')
    record = {
        'ticker': df.iloc[0,0],
        'exchange': df.iloc[0,1],
        'time_series_data': time_series_data
    }
    financial_collection.insert_one(record)


def delete_all_documents():
    financial_collection.delete_many({})
    
# Function to query all financial data in financial_collection collection
def get_financial_data():
    cursor = financial_collection.find()
    return pd.DataFrame(list(cursor))
