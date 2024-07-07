import numpy as np
import random
import pandas as pd
from ML_Models.markowitz import *
import fitz  # Import de PyMuPDF
import datetime as dt
from db_connection import *
import os

def calculate_return_asset(df_asset, stocks):
    _return = pd.DataFrame()
    for stock in stocks:
        _return[stock] = np.log(df_asset[stock] / df_asset[stock].shift(1)) 
        _return[stock] = 1 + _return[stock]
        _return[stock] = _return[stock].drop(0).reset_index(drop=True)
    _return.dropna(inplace=True)
    return _return

def calculate_contrib(perf, allocation):
    return perf * allocation

def total_perf(contrib_array):
    return np.sum(contrib_array)

def total_perf_stocks(stocks, _return):
    total_perf = pd.DataFrame()
    for stock in stocks:
        perf = np.cumprod(_return[stock])
        total_perf[stock] = perf
    return total_perf

# Function to convert Google Drive URL to direct download URL
def get_direct_gdrive_url(view_url):
    file_id = view_url.split('/')[-2]
    return f'https://drive.google.com/uc?id={file_id}'

# Function to download a file from a URL
def download_file(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful

    with open(save_path, 'wb') as file:
        file.write(response.content)
def simulation(list_assets):

    """
    INIT PDF EDITION
    """

    doc = fitz.open()  # Créer un nouveau document PDF
    page = doc.new_page()  # Ajouter une nouvelle page
    page.insert_text((200, 50), f"MLMarketInsight\nSimulation Report\n{dt.date.today()}", fontname='helv', fontsize=25)
    page.insert_text((50, 150), f"MLMarketInsights is a financial solution aiming to democratize machine learning and quantitative\nanalysis. It aims to bring efficient and precise analysis of future performance \nof a given portfolio, to enlighten investors' investment decisions. MLMarketInsights doesn't manage funds.")


    """
    INIT STOCKS AND ALLOCATION 100% / n
    """

    # stocks = np.array(["AAPL", "MSFT", "NVDA", "TSLA"])
    stocks = np.array(list_assets)
    alloc = [(1 / len(stocks)) for i in range(len(stocks))]


    """
    INSERT INIT ASSET AND ALLOCATION TO PDF
    """
    page.insert_text((50, 220), f"Number of asset : {len(stocks)}")
    page.insert_text((50, 240), f"Asset : {str(stocks)}")
    page.insert_text((50, 260), f"Allocation initiale par asset : {round((1 / len(stocks))*100, 3)}%")


    """
    FETCH HISTORICAL DATA BDD
    """
    financial_data_fetched = get_financial_data()
    asset_to_evaluate = financial_data_fetched[financial_data_fetched["ticker"].isin(stocks)].reset_index(drop=True)

    """
    DATAFRAME AVEC UNIQUEMENT LA COLONNE CLOSE POUR CHAQUE ASSET
    """
    dataset = pd.DataFrame()
    for i in range(len(asset_to_evaluate["ticker"])):
        asset = asset_to_evaluate.loc[i, "ticker"]
        time_series = pd.DataFrame(asset_to_evaluate.loc[i, "time_series_data"])
        dataset[asset] = time_series["close"]

    dataset["date"] = pd.DataFrame(asset_to_evaluate.loc[0, "time_series_data"]).loc[:,"date"]
    dataset.set_index("date",inplace=True)
    new_dataset = dataset[::-1]


    """
    PLOT HISTORICAL FINANCIAL DATA
    """
    show_data(new_dataset, page, 270)


    """
    CALCULATE LOG RETURN OF CLOSE VALUE
    """
    logReturn = calculate_return(new_dataset)


    """    
    DISPLAYING MEAN VALUES OF RETURNS
    """    
    show_statistics(logReturn)

    """
    DISPLAYING EXPECTED PORTFOLIO RETURN / EXPECTED PORTFOLIO RISK (VOLATILITY)
    """
    show_mean_variance(logReturn, alloc, page)





    """
    MARKOWITZ PORTFOLIO OPTIMIZATION
    """


    """
    GENERATING RANDOM PORTFOLIO FOR TESTING
    """

    pweights, means, risks = generate_portfolio(logReturn, stocks)


    """
    EFFICIENT FRONTIER AND SHARPE RATIO
    """
    #show_portfolio(means, risks)
    optimum = optimize_portfolio(alloc, logReturn, stocks)

    """
    OPTIMUM PORTFOLIO 
    """
    page, opt_portfolio, stat = print_optimal_portfolio(optimum, logReturn, page, stocks)

    page = doc.new_page()  # Ajouter une nouvelle page
    page.insert_text((200, 50), f"MLMarketInsight\nSimulation Report\n{dt.date.today()}", fontname='helv', fontsize=25)
    page.insert_text((50, 150), f"MLMarketInsights is a financial solution aiming to democratize machine learning and quantitative\nanalysis. It aims to bring efficient and precise analysis of future performance \nof a given portfolio, to enlighten investors' investment decisions. MLMarketInsights doesn't manage funds.")

    show_optimal_portfolio(optimum, logReturn, means, risks, page)


    """
    PREDICTION
    """
    

    """
    FETCH PREDICTION DATA 
    """
    prediction = get_prediction_data()
    asset_to_evaluate = prediction[prediction["ticker"].isin(stocks)].reset_index(drop=True)
    pred = pd.DataFrame()

    for i in range(len(asset_to_evaluate["ticker"])):
        asset = asset_to_evaluate.loc[i, "ticker"]
        time_series = pd.DataFrame(asset_to_evaluate.loc[i, "time_series_data"])
        pred[asset] = time_series["close"]
    

    """
    CALCULATE LOG RETURN OF PREDICTIONS
    """
    _return = calculate_return_asset(pred, stocks)

    
    """
    CALCULATE TOTAL PERFORMANCE
    """
    total_perf_per_asset = total_perf_stocks(stocks, _return)


    """
    GET FINAL PERFORMANCE OF ASSET (CUMULATIVE PRODUCT OF RETURN TO THE LAST PREDICTION DAY)
    """
    perf_final = total_perf_per_asset.iloc[len(total_perf_per_asset) - 1, :]
    print(perf_final)
    print("OPTIMAL PORTFOLIO / ")
    print(opt_portfolio)


    """
    CONTRIBUTION TO PERFORMANCE (allocation markowitz * performance)
    """
    contrib = calculate_contrib(perf_final, opt_portfolio) #contrib totale par asset : variable
    contrib_time_series = calculate_contrib(total_perf_per_asset, opt_portfolio) #contrib totale par asset : time_series



    """
    INSERT DATA ABOUT PREDICTION
    """
    
    for i in range(len(stocks)):
        page = doc.new_page()  # Ajouter une nouvelle page
        page.insert_text((200, 50), f"MLMarketInsight\nSimulation Report\n{dt.date.today()}", fontname='helv', fontsize=25)
        page.insert_text((50, 150), f"MLMarketInsights is a financial solution aiming to democratize machine learning and quantitative\nanalysis. It aims to bring efficient and precise analysis of future performance \nof a given portfolio, to enlighten investors' investment decisions. MLMarketInsights doesn't manage funds.")

        page.insert_text((50, 240), f"DETAILED SUMMARY OF ASSETS IN PORTFOLIO\n{stocks[i]}", fontname='helv', fontsize=15)
        page.insert_text((50, 280), f"Expected performance of {stocks[i]} : {round((perf_final[i] - 1)*100, 3)}%")
        page.insert_text((50, 300), f"Contribution à la performance : {round((contrib[i])*100, 3)}%")
        page.insert_text((50, 320), f"asset 1 graph prediction FETCH PHOTO")
        asset_to_evaluate = prediction[prediction["ticker"].isin(stocks)].reset_index(drop=True)


        pred_file = 'pred_file.png'


        asset = asset_to_evaluate[asset_to_evaluate["ticker"] == stocks[i]]
        path = asset["img_prev"].iloc[0]
        direct_url = get_direct_gdrive_url(path)  # Convert to direct download URL

        download_file(direct_url, pred_file)
        # Définir la position et la taille de l'image sur la page
        img_height = 200  # Par exemple
        img_width = 500  # Par exemple
        rect = fitz.Rect(50, 330, 50 + img_width, 330 + img_height)
        
        # Insert the image into the page
        page.insert_image(rect, filename=str(pred_file))
        os.remove(pred_file)
    

 

    """
    CALCUL PORTFOLIO PERFORMANCE
    """
    perf_portefeuille_totale = np.sum(contrib) # contrib totale portefeuille : variable
    perf_portefeuille_time_series = contrib_time_series.sum(axis=1) #contrib totale portefeuille: time_series


    """
    INSERT PREDICTION INFORMATION : OPTIMUM PORTFOLIO, PERF FINAL, ALLOCATION
    """
    page = doc.new_page()  # Ajouter une nouvelle page
    page.insert_text((200, 50), f"MLMarketInsight\nSimulation Report\n{dt.date.today()}", fontname='helv', fontsize=25)
    page.insert_text((50, 150), f"MLMarketInsights is a financial solution aiming to democratize machine learning and quantitative\nanalysis. It aims to bring efficient and precise analysis of future performance \nof a given portfolio, to enlighten investors' investment decisions. MLMarketInsights doesn't manage funds.")

    page.insert_text((50, 240), f"DETAILED SUMMARY OF PORTFOLIO", fontname='helv', fontsize=15)
    page.insert_text((50, 260), f"Performance of portfolio : {round((perf_portefeuille_totale - 1)*100, 3)}%", fontname='helv', fontsize=15)
    show_data(perf_portefeuille_time_series, page, 260)
    #signature pour pdf
    signature = pd.Timestamp.now()


    signature = str(signature).split(" ")
    timestamp = signature[1].replace(":", "-").replace(".", "-")
    
    # Générer le nom de fichier avec un timestamp valide
    filename = f"MLMarketInsight_Report.pdf"
    
    doc.save(filename)
    doc.close()   

    return {'pdf_code' :str(signature[0]+timestamp), "rendement": perf_portefeuille_totale, 'assets': ', '.join(stocks), 'allocation':  ', '.join(map(str, opt_portfolio))} 



if __name__ == '__main__':
    
    list_user_assets = ["AAPL", "MSFT", "NVDA", "TSLA"]
    simulation(list_user_assets)