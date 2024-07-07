
from packages_import import *

client = MongoClient('mongodb+srv://matthieuvichet:aO0mNOu20DVJpKHz@mlmarketinsights.vgtptz0.mongodb.net/')
db = client['MLMarketInsights']
financial_collection = db['FinancialData'] # creer le mien pour save les predict et ticker, erreur image
prevision_collection = db['PrevisionData']

def insert_financial_data(df):
    """
    Information inserted comes from the API
    """
    time_series_data = df.drop(columns=['ticker', 'exchange']).to_dict('records')
    record = {
        'ticker': df.iloc[0,0],
        'exchange': df.iloc[0,1],
        'time_series_data': time_series_data
    }
    financial_collection.insert_one(record)


def get_shareable_link(ticker, json_file_path='CODE/shareable_links.json'):
    with open(json_file_path, 'r') as file:
        data = json.load(file) 
    ticker_filename = f"{ticker}_PREV.png"
    for item in data:
        if item[0] == ticker_filename:
            return item[1]
    
    return None

def insert_prevision_data(df, confidence_level): 
    """
    Information comes from code and not the API
    """
    time_series_data = df.drop(columns=['ticker']).to_dict('records')
    ticker = df.iloc[0,0]
    record = {
        'ticker': ticker,
        'time_series_data': time_series_data,
        'confidence_level' : confidence_level,
        'img_prev': get_shareable_link(ticker),
        'img_ticker': get_shareable_link(ticker, "CODE/asset_icons.json")
    }
    prevision_collection.insert_one(record)


def delete_all_documents():
    financial_collection.delete_many({})

def delete_prevision_documents():
    prevision_collection.delete_many({})
    
# Function to query financial data
def get_financial_data():
    cursor = financial_collection.find()
    return pd.DataFrame(list(cursor))

def get_prevision_data():
    cursor = prevision_collection.find()
    return pd.DataFrame(list(cursor))