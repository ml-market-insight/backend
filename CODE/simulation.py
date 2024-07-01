import numpy as np
import random
import pandas as pd
from ML_Models.markowitz import *
import fitz  # Import de PyMuPDF
import datetime as dt
from db_connection import *

asset_db = ["EURUSD", 'USDJPY', 'AAPL', "ABC", "BCDG", "fg", "PZJD", "PDJC", "PEJD", "oickc", "fgh", "po", "jcikek", "pdiod"]

def calculate_return_asset(df_asset, stocks):
    _return = pd.DataFrame()
    print(df_asset.shape)
    for stock in stocks:
        _return[stock] = np.log(df_asset[stock] / df_asset[stock].shift(1)) 
        _return[stock] = 1 + _return[stock]
        print(_return[stock])
        _return[stock] = _return[stock].drop(0).reset_index(drop=True)
    _return.dropna(inplace=True)
    return _return

def calculate_contrib(perf, allocation):
    return perf * allocation

def total_perf(contrib_array):
    return np.sum(contrib_array)

def simulation():

    """
    INIT PDF EDITION
    """

    doc = fitz.open()  # Créer un nouveau document PDF
    page = doc.new_page()  # Ajouter une nouvelle page
    page.insert_text((200, 50), f"MLMarketInsight\nSimulation Report\n{dt.date.today()}", fontname='helv', fontsize=25)
    page.insert_text((50, 150), f"MLMarketInsights is a financial solution aiming to democratize machine learning and quantitative\nanalysis. It aims to bring efficient and precise analysis of future performance \nof a given portfolio, to enlighten investors' investment decisions. MLMarketInsights doesn't manage funds.")

    #nb_asset = input("enter the number of asset")

    stocks = np.array(["AAPL", "MSFT", "NVDA"])

    page.insert_text((50, 220), f"Number of asset : {len(stocks)}")
    initial_alloc = 1 / len(stocks)



    """
    CREATION ALLOCATION INITIALE 100% / n
    """
    alloc = [initial_alloc for i in range(len(stocks))]


    """
    FETCH HISTORICAL DATA BDD
    """
    dataset2 = get_financial_data()
    to_evaluate = dataset2[dataset2["ticker"].isin(stocks)].reset_index(drop=True)
    test = pd.DataFrame(to_evaluate.loc[0, "time_series_data"])
    dataset = pd.DataFrame()

    """
    DATAFRAME AVEC UNIQUEMENT LA COLONNE CLOSE POUR CHAQUE ASSET
    """
    for i in range(len(to_evaluate["ticker"])):
        asset = to_evaluate.loc[i, "ticker"]
        time_series = pd.DataFrame(to_evaluate.loc[i, "time_series_data"])
        dataset[asset] = time_series["close"]

    dataset["date"] = pd.DataFrame(to_evaluate.loc[0, "time_series_data"]).loc[:,"date"]
    print(dataset)
    dataset.set_index("date",inplace=True)
    print(dataset)
    new_dataset = dataset[::-1]


    """
    PLOT HISTORICAL FINANCIAL DATA
    """
    show_data(new_dataset)


    """
    CALCULATE LOG RETURN
    """
    # CALCULATING THE ANNUAL RETURN OF ASSETS
    logReturn = calculate_return(new_dataset)

    # DISPLAYING MEAN VALUES OF RETURNS
    show_statistics(logReturn)

    # DISPLAYING EXPECTED PORTFOLIO RETURN / EXPECTED PORTFOLIO RISK (VOLATILITY)
    show_mean_variance(logReturn, alloc)


    """
    MARKOWITZ PORTFOLIO OPTIMIZATION
    """
    # GENERATING RANDOM PORTFOLIO FOR TESTING
    pweights, means, risks = generate_portfolio(logReturn, stocks)

    # EFFICIENT FRONTIER AND SHARPE RATIO
    show_portfolio(means, risks)
    optimum = optimize_portfolio(alloc, logReturn, stocks)

    # Afficher le portefeuille optimal et ses statistiques
    page, opt_portfolio, stat = print_optimal_portfolio(optimum, logReturn, page)
    show_optimal_portfolio(optimum, logReturn, means, risks, page)

    page.insert_text((50, 500), f"Volatility expected : (int) or None")
    page.insert_text((50, 520), f"Return expected : (int) or None")
    page.insert_text((50, 540), f"Sharpe ratio expected : (int) or None")
    page.insert_text((50, 560), f"Graph fromage portfolio / allocation : (int) or None")



    """
    PREDICTION
    """
    
    page = doc.new_page()  # Ajouter une nouvelle page
    page.insert_text((200, 50), f"MLMarketInsight\nSimulation Report\n{dt.date.today()}", fontname='helv', fontsize=25)

    page.insert_text((50, 150), f"ASSET IN PORTFOLIO DETAIL", fontname='helv', fontsize=15)
    page.insert_text((50, 170), f"expected performance of this asset")
    page.insert_text((50, 190), f"contribution à la performance")
    page.insert_text((50, 210), f"asset 1 graph prediction ")


    """
    FETCH PREDICTION DATA 
    """
    prediction = get_prediction_data()
    to_evaluate = prediction[prediction["ticker"].isin(stocks)].reset_index(drop=True)
    pred = pd.DataFrame()

    for i in range(len(to_evaluate["ticker"])):
        asset = to_evaluate.loc[i, "ticker"]
        time_series = pd.DataFrame(to_evaluate.loc[i, "time_series_data"])
        pred[asset] = time_series["close"]
    

    """
    CALCULATE LOG RETURN OF PREDICTIONS
    """
    test = calculate_return_asset(pred, stocks)
    print(test)
    total_perf = pd.DataFrame()
    for stock in stocks:
        perf = np.cumprod(test[stock])
        print(perf)
        total_perf[stock] = perf
    print(total_perf)

    """
    GET FINAL PERFORMANCE OF ASSET
    """
    perf_final = total_perf.iloc[len(total_perf) - 1, :]
    print(perf_final)
    print("OPTIMAL PORTFOLIO / ")
    print(opt_portfolio)
    print(alloc)

    """
    CONTRIBUTION TO PERFORMANCE (allocation markowitz * performance)
    """
    contrib = calculate_contrib(perf_final, opt_portfolio) #contrib totale par asset : variable
    contrib_time_series = calculate_contrib(total_perf, opt_portfolio) #contrib totale par asset : time_series
    print(contrib)



    perf_portefeuille_totale = np.sum(contrib) # contrib totale portefeuille : variable
    print(perf_portefeuille_totale)
    perf_portefeuille_time_series = contrib_time_series.sum(axis=1) #contrib totale portefeuille: time_series
    print(perf_portefeuille_time_series)


    #signature pour pdf
    signature = pd.Timestamp.now()

    print(signature)
    


    















if __name__ == '__main__':

    simulation()