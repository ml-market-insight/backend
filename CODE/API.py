
from packages_import import *
from db_connection import *

def fetch_data_with_retry(func, *args, retries=6, delay=1):
    """
    func : fonction à appeler pour récupérer les données
    *args: arguments positionnels de la fonction func
    retries  : nb de fois où on relance la fonction func
    delay : délais initial de une seconde
    """
    global key_index
    
    for attempt in range(retries):
        try:
            return func(*args, keys[key_index])
        except Exception as e:
            if e.code == 429: # erreur 429 : erreur de type "Trop de demandes" i.e on a dépassé le nombre de calls sur cette clef
                print(f"Rate limit exceeded with key {keys[key_index]}. Retrying with next key...")
                key_index = (key_index + 1) % len(keys)  # Increment key index and wrap around if necessary
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
    raise Exception(f"Failed to fetch data after {retries} retries with all keys")

def get_jsonparsed_data(url):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    return json.loads(data)

def get_nasdaq_ticker(key):
    NASDAQ_TICKERS = (f"https://financialmodelingprep.com/api/v3/symbol/NASDAQ?apikey={key}")
    nasdaq_json =get_jsonparsed_data(NASDAQ_TICKERS)
    nasdaq_df = pd.DataFrame(nasdaq_json)
    return nasdaq_df.sort_values(by='marketCap', ascending=False).head(50).loc[:, ["symbol", "exchange"]]

def get_forex_ticker(key):
    FOREX_TICKER = (f"https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs?apikey={key}")
    forex_json =get_jsonparsed_data(FOREX_TICKER)
    forex_df = pd.DataFrame(forex_json)
    return forex_df[(forex_df['symbol'].str.startswith(('EUR', 'CHF', 'JPY', 'AUD', 'CAD', 'USD'))) & (forex_df['currency'].isin(['AUD', 'CHF', 'CAD', 'JPY', 'USD']))][['symbol']]

def get_historical_price(ticker, key):
    historical_price_url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={key}")
    historical_price_json =get_jsonparsed_data(historical_price_url)
    historical_price_df = pd.DataFrame(historical_price_json)
    return historical_price_df

def get_indicator(ticker, indicator, key):
    indicateur_url = f"https://financialmodelingprep.com/api/v3/technical_indicator/1day/{ticker}?type={indicator}&period=10&apikey={key}"
    indicateur_json =get_jsonparsed_data(indicateur_url)
    indicateur_df = pd.DataFrame(indicateur_json)
    return indicateur_df


def populate_BDD(assets_array):
    indicators = ["ema", "dema", "williams", "rsi", "adx", "standardDeviation"]

    for i in range(0,len(assets_array)):
        historical_price = fetch_data_with_retry(get_historical_price,assets_array.iloc[i, 0])
        historical_data = historical_price['historical'].apply(pd.Series)[['date', 'close', 'volume','high', 'low']]
        for indicator in indicators:
            historical_data[indicator] = fetch_data_with_retry(get_indicator,assets_array.iloc[i, 0], indicator).loc[:,indicator]

        historical_data.drop(index=historical_data.index[-1], inplace=True)
        historical_data["ticker"] = assets_array.iloc[i, 0]
        historical_data["exchange"] = assets_array.iloc[i, 1]

        column_order = ["ticker", "exchange", "date", "close", 'high', 'low' , "volume", "ema", "dema", "williams", "rsi", "adx", "standardDeviation"]
        historical_data = historical_data[column_order]
        insert_financial_data(historical_data) 

def populate_BDD_with_prevision():

    niv_confiance_df = pd.DataFrame(pd.read_csv(rf"CODE\ML_Models\random_forest_csv\previsions\confidence_level.csv")).drop(columns = "Unnamed: 0", axis = 1 )
    all_asset_prevision_df = pd.DataFrame(pd.read_csv(rf"CODE\ML_Models\random_forest_csv\previsions\prediction_df.csv")).drop(columns = "Unnamed: 0", axis = 1 )


    for ticker, group_df in all_asset_prevision_df.groupby('ticker'):
        group_df = group_df.reset_index(drop=True)
        insert_prevision_data(df = group_df, confidence_level = niv_confiance_df[niv_confiance_df["ticker"] == ticker].iloc[0,1] )


def BDD_adding_data(historical:bool = False, prevision:bool = True, tickerFullName:bool = False):
    """
    Theoricaly not both at the same time
    """
    if historical : 
        print("deleting")
        delete_all_documents()
        print("deleted")
        time.sleep(5)

        print("Fetching data using financialmodelingprep API...")
        nasdaq_ticker_array, forex_ticker_array = fetch_data_with_retry(get_nasdaq_ticker), fetch_data_with_retry(get_forex_ticker)
        asset_array = pd.concat([nasdaq_ticker_array, forex_ticker_array])
        asset_array.fillna("FOREX", inplace=True)

        print(asset_array)
        print("Populating database...")

        populate_BDD(asset_array)
        print("Successfully populated database...")
    
    if prevision : 
        print("deleting")
        delete_prevision_documents()
        print("deleted")
        time.sleep(5)

        print("Populating database with previsions...")
        populate_BDD_with_prevision()
        print("Successfully populated database with prevision...")

    if tickerFullName : 
        print("deleting tickerFullName")
        delete_asset_full_name_documents()
        print("deleted tickerFullName")
        time.sleep(5)

        print("Populating database with tickerFullName...")
        insert_full_name_data()
        print("Successfully populated database with tickerFullName...")


if __name__ == "__main__":
    BDD_adding_data(historical = False, prevision = False, tickerFullName = True)

