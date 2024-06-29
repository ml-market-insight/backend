print(">>> Import packages...")
from packages_import import *
from db_connection import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV , cross_val_predict, KFold
import seaborn as sns
import plotly.express as px
import math
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from sklearn.linear_model import LinearRegression



print(">>> Import packages done !")


def get_full_data_df(dataset, tickers:list): # TODO : check
    """
    Dataset with each asset information
    Return a dataframe containing each asset years of data
    """
    print(">>> get_full_stock_df() ... ")
    # tickers = [stock for stock in dataset["ticker"] ]
    dataframe = pd.DataFrame( columns = list(dataset.time_series_data.iloc[0][0].keys()) )
    for stock in tickers : 
        dataset_one_stock = dataset[dataset["ticker"].isin([stock])].reset_index(drop=True)
        one_stock_df = pd.DataFrame(dataset_one_stock.loc[0, "time_series_data"])
        one_stock_df["ticker"] = dataset_one_stock.loc[0,"ticker"]
        dataframe = pd.concat([dataframe, one_stock_df])
    # dataframe = dataframe[::-1].reset_index(drop = True) # ordre décroissant des dates de haut en bas
    return dataframe


def get_one_data_df(dataset, stock): # TODO : check
    """
    stock : example ["AAPL"] i.e doit être de la forme ["nom_du_stock"]
    Retourne un dataframe d'années de data pour un stock donné
    """
    print(">>> get_one_stock_df()...")
    dataset_one_stock = dataset[dataset["ticker"].isin(stock)].reset_index(drop=True)
    dataframe = pd.DataFrame(dataset_one_stock.loc[0, "time_series_data"])
    dataframe["ticker"] = dataset_one_stock.loc[0, "ticker"]
    dataframe = dataframe[::-1].reset_index(drop = True)
    return dataframe



def reformat_df(dataframe): # TODO : check
    """
    Function : Ajout des colonnes year, month et day puis réordonner les colonnes
    """

    print(">>> reformat_df() ... ")
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["day"] = dataframe["date"].dt.day

    cols = dataframe.columns.tolist()
    cols_to_move = ["ticker", "date", "year", "month", "day"]
    new_cols_order = cols_to_move + [col for col in cols if col not in cols_to_move]
    final_df = dataframe[new_cols_order]

    return final_df



def dataframe_analysis_1(dataframe): # TODO : check
    """    
    This function purpose is only for machine learning on historical data
    """
    df = dataframe[[ col for col in dataframe.columns.tolist() if col not in [ "date" , "year" , "month" , "day"] ]]

    print(f"Dataframe.head():\n", dataframe.head())
    print(f"Dataframe.info():\n", dataframe.info())
    print(f"Dataframe.describe():\n", df.describe())
    print(f"Dataframe.isnull().sum(), do we have null data:\n", dataframe.isnull().sum())


 
def dataframe_analysis_2(dataframe, variable:str):# TODO : check
    """
    This function purpose is only for machine learning on historical data
    variable : une seul colonne à analyser
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=dataframe[variable])
    plt.title(f"{variable.upper()} Boxplot")

    plt.subplot(1, 2, 2)
    sns.histplot(dataframe[variable], kde=True)
    plt.title(f"{variable.upper()} Histogramme")

    plt.tight_layout()
    plt.show()

    # Tracé de base (plot)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=dataframe, x=dataframe.index, y=variable) 
    plt.title(f"{variable.upper()} Plot")
    plt.show()



def matrice_correlation(dataframe) :  # TODO : check
    """    
    This function purpose is only for machine learning on historical data
    """
    df = dataframe[[col for col in dataframe.columns if col not in ['ticker']]]
    plt.figure(figsize=(15,10))
    matrice = df.corr()
    sns.heatmap(matrice,annot=True)
    plt.title('Correlation')


def avg_close_price(dataframe): # TODO : check
    # TODO : REVOIR LA FENETRE POUR LA PREVISION DES CLOSE FUTUR
    df = dataframe.copy()
    df['Avg_close'] = df['close'].rolling(window=5, min_periods=1).mean()
    return df


def ml_data_preparation(dataframe): # TODO : check

    print(f">>> ml_data_preparation() ...")
    # One-hot encoding : Transformer des données catégoriques (=string) en valeurs numériques
    dataframe = pd.get_dummies(dataframe)
    # variable à prédire : close variable
    labels = np.array(dataframe["close"])
    # drop la variable à prédire
    dataframe = dataframe.drop(columns = "close" , axis = 1)
    # sauvegarde des colonnes pour plus tard : 
    features_list = dataframe.columns.tolist()
    features = np.array(dataframe)

    # sépare de manière aléatoire le df en training et test set
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42, shuffle = False, stratify= None)
    # test_size = 0.25 <= 25% des données seront utilisées pour les test
    # random_state = 42 <= division de données identiques à chaque fois que je run le code ==> résultats reproductibles. 42 est une graine pour le générateur
    # sa valeur n'a pas d'importance en soi, il faut juste que ce soit le même nombre à chaque fois
    #  shuffle = False, stratify= None car on a des séries temporelles. C'et important de maintenir l'ordre temporel !!!! Entraînement = vieille data, Test = Récentes
    """
    RAPPEL : les données sont mélangées avant d'être divisées en ensemble d'entraînement et de test. Mélangés car si divisées tel quel alors chaque ensemble aura
                des données appartenant au même interval de date
    """
    train_test_dict = {"train_features" : train_features,
                        "test_features" : test_features, 
                        "train_labels" : train_labels, 
                        "test_labels" : test_labels
                        }
    
    print('Training Features Shape:', train_test_dict["train_features"].shape) # on a bien les données anciennes dans le train 
    print('Training Labels Shape:', train_test_dict["train_labels"].shape) # on a bien les données anciennes dans le train
    print('Testing Features Shape:', train_test_dict["test_features"].shape) # on a bien les données nouvelles dans le test
    print('Testing Labels Shape:', train_test_dict["test_labels"].shape) # on a bien les données nouvelles dans le test
    
    return labels, dataframe, features, features_list , train_test_dict
    



def ml_model_baseline(dict_X_Y, list_features): # TODO : check
    test_features = dict_X_Y["test_features"]
    baseline_preds = test_features[:, list_features.index('Avg_close')]
    # Baseline errors
    baseline_errors = abs(baseline_preds - dict_X_Y["test_labels"])
    avg_baseline_error = round(np.mean(baseline_errors), 2)
    print('Average baseline error in degree : ', avg_baseline_error )
    return avg_baseline_error




def RFR_model_1(dict, params, ticker:str): # TODO : check
    # Train the model on the training data : 
        # Instancier le modèle
    rf = RandomForestRegressor(**params , random_state = 42)
        # Train du modèle
    rf.fit(dict["train_features"], dict["train_labels"])
        # Use the forest's predict method on the test data
    predictions = rf.predict(dict["test_features"])
    predictions_rounded = np.round(predictions, 2)
    joblib.dump(rf, rf'CODE\ML_Models\random_forest_csv\models\{ticker}_random_forest_model.pkl')

        # Calculate the absolute errors
    errors = abs(predictions_rounded - dict["test_labels"])

    mean_actual = np.mean(dict["test_labels"])
    mean_predict = np.mean(predictions)
    mean_absolute_error = np.mean(errors)

    confidence_level = 100 - round(np.mean(errors), 2)
    
        # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(mean_absolute_error, 2), 'degrees.')
        # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / dict["test_labels"])
        # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    r2 = r2_score(dict['test_labels'], predictions_rounded)
    mse = mean_squared_error(dict['test_labels'], predictions_rounded)
    rmse = np.sqrt(mse)
    print(f"==== {ticker.upper()} ==== ")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(predictions_rounded, label='Prédictions', marker='o')
    plt.plot(dict["test_labels"], label='Valeurs réelles', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Valeur')
    plt.title(f'Prédictions vs Valeurs réelles pour {ticker}')
    plt.legend()
    output_path = rf"CODE\ML_Models\random_forest_csv\Images\{ticker}_histo_vs_predict.png"
    plt.savefig(output_path)
    plt.tight_layout()
    # plt.show()
    plt.close()

    return confidence_level


def RFR_model_2(dict): # TODO : check
    # USELESS FUNCTION NEVER USED
    X_train = dict['train_features']
    y_train = dict['train_labels']
    X_test = dict['test_features']
    y_test = dict['test_labels']

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)


def grid_search_random_forest(train_features, train_labels, test_features, test_labels, param_grid, path): # TODO : CHECK
    print("GRID SEARCH RANDOM FOREST...")
    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    
    # Set up the GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2', return_train_score=True)
    print("WORKING....")
    # Fit the grid search
    grid_search.fit(train_features, train_labels)
    print("FITTING....")
    # Extract results
    results = grid_search.cv_results_
    print("EXTRACTING....")
    # Create a DataFrame to store the parameters and metrics
    df_results = pd.DataFrame(results)
    
    # Add columns for metrics
    df_results['test_r2'] = np.nan
    df_results['test_mse'] = np.nan
    df_results['test_rmse'] = np.nan
    df_results['cv_r2'] = np.nan
    df_results['cv_mse'] = np.nan
    df_results['cv_rmse'] = np.nan
    
    # Initialize lists to store metrics
    r2_list = []
    mse_list = []
    rmse_list = []
    cv_r2_list = []
    cv_mse_list = []
    cv_rmse_list = []
    
    # Create a KFold object for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate metrics for each iteration
    for idx, params in enumerate(grid_search.cv_results_['params']):
        print("WORKING 2....")
        # Train the model with the current parameters
        rf.set_params(**params)
        print("FITTING2....")
        rf.fit(train_features, train_labels)
        print("PREDICT....")
        # Predict on the test set
        y_pred = rf.predict(test_features)
        
        # Calculate metrics
        r2 = r2_score(test_labels, y_pred)
        mse = mean_squared_error(test_labels, y_pred)
        rmse = np.sqrt(mse)
        
        # Append metrics to lists
        r2_list.append(r2)
        mse_list.append(mse)
        rmse_list.append(rmse)
        print("CROSS VALID....")
        # Perform cross-validation
        cv_preds = cross_val_predict(rf, train_features, train_labels, cv=kf)
        
        # Calculate cross-validation metrics
        cv_r2 = r2_score(train_labels, cv_preds)
        cv_mse = mean_squared_error(train_labels, cv_preds)
        cv_rmse = np.sqrt(cv_mse)
        
        # Append cross-validation metrics to lists
        cv_r2_list.append(cv_r2)
        cv_mse_list.append(cv_mse)
        cv_rmse_list.append(cv_rmse)
    
    # Add metrics to the DataFrame
    df_results.loc[:, 'test_r2'] = r2_list
    df_results.loc[:, 'test_mse'] = mse_list
    df_results.loc[:, 'test_rmse'] = rmse_list
    df_results.loc[:, 'cv_r2'] = cv_r2_list
    df_results.loc[:, 'cv_mse'] = cv_mse_list
    df_results.loc[:, 'cv_rmse'] = cv_rmse_list
    
    # Add best parameters to the DataFrame
    df_results['best_params'] = str(grid_search.best_params_)
    
    # Save the DataFrame to a CSV file
    df_results.to_csv(path + r"\Test.csv", index=False)
    
    return df_results

def format_df(dataset, stock:str):
    """
    prepare the one_stock_df
    """
    df = get_one_data_df(dataset2, [stock])
    if 'high' in df.columns :
        df = df.drop(["high"], axis = 1)
    if 'low' in df.columns :
        df = df.drop(["low"], axis = 1)

    df = reformat_df(df)
    df = avg_close_price(df)

    return  df

def create_train_model(asset_df, asset_name, best_params, analysis:bool = False) : 
    """
    Function to do the ML training and testing
    dataset : dataset from BDD
    best_params : best parameters founded in GridCV function 
    list_tickers : list of tickers <== 75 tickers in that list
    analysis : (Default False) wether to show the plot or not. Used for historical data ML
    """


    if analysis :
        exclude_columns = ['ticker', 'date', 'year', 'month', 'day']
        dataframe_analysis_1(asset_df)
        for col in [c for c in asset_df.columns if c not in exclude_columns] : 
            dataframe_analysis_2(asset_df, col)
            pass
        matrice_correlation(asset_df)
        
    """
    State the question and determine required data : Prédire l'évolution de la variable Close
    Acquire the data in an accessible format : C'est fait, il manque aucune data dans one_stock pour AAPL et il en manque 19 sur tous les indicateurs dans full_stock_df
    Identify and correct missing data points/anomalies as required : Aucune dans one_stock mais faire un code pour trouver et gérer
    """
    # Prepare the data for the machine learning model :
    asset_df = asset_df[[col for col in asset_df.columns.tolist() if col != "date"]]
    label, dataframe, features, features_list , dict_X_Y = ml_data_preparation(asset_df)

    # Establish a baseline model that you aim to exceed :
    avg_baseline_error = ml_model_baseline(dict_X_Y, features_list)

    """
    n_estimators : Nombre d'arbres dans la forêt : Typiquement entre 100 et 1000
    max_depth : Profondeur maximale des arbres. Peut être None (pour une profondeur illimitée) ou des valeurs comme 10, 20, 30.
    min_samples_split : Nombre minimum d'échantillons requis pour diviser un nœud interne. Entre 2 et 10.
    min_samples_leaf : Nombre minimum d'échantillons requis pour être à un nœud terminal (feuille). Entre 1 et 10
    max_features : Nombre de caractéristiques à considérer lors de la recherche de la meilleure division.  Peut être une fraction comme 0.5, sqrt ou log2
    """
    conf_level = RFR_model_1(dict_X_Y , best_params, asset_name)

    return conf_level



def generate_dataframe(start_date, num_days:int = 10): # TODO : check

    columns = ['ticker', 'date', 'year', 'month', 'day', 'close', 'volume', 'ema', 'dema',
               'williams', 'rsi', 'adx', 'standardDeviation', 'Avg_close']
    
    nyse = mcal.get_calendar('NYSE')
    end_date = start_date + timedelta(days=num_days * 2)  # Extra days to account for weekends and holidays
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = mcal.date_range(schedule, frequency='1D')
    trading_days = trading_days[:num_days]
    data = {
        'date': [day.strftime('%Y-%m-%d') for day in trading_days],
        'year': [day.year for day in trading_days],
        'month': [day.month for day in trading_days],
        'day': [day.day for day in trading_days]
    }
    
    df = pd.DataFrame(data, columns=columns)
    return df


def create_features(asset_df, ticker: str, num_days = 30): # TODO : ENLEVER

    for var in ['high', 'low','volume', 'ema', 'dema', 'williams', 'rsi', 'adx', 'standardDeviation','Avg_close'] :
        df_train = asset_df.dropna()
        df_predict = asset_df[asset_df[var].isna()]

        # Définir les variables explicatives et cibles pour la régression linéaire
        X_train = df_train[['date']].values.reshape(-1, 1)
        y_high_train = df_train['high'].values
        y_low_train = df_train['low'].values

        # Entraîner deux modèles de régression linéaire pour 'high' et 'low'
        model_high = LinearRegression()
        model_high.fit(X_train, y_high_train)

        model_low = LinearRegression()
        model_low.fit(X_train, y_low_train)

        # Prédire les valeurs manquantes dans df_predict
        X_predict = df_predict[['date']].values.reshape(-1, 1)

        df_predict['high'] = model_high.predict(X_predict)
        df_predict['low'] = model_low.predict(X_predict)

        # Remplacer les NaN dans df par les valeurs prédites
        asset_df.update(df_predict)


    a =True
    # Calcul high and low -----------------------------------------------------------
    for index in range(asset_df[asset_df['close'].isna()].index[0], len(asset_df)):
        
        current_ticker = asset_df.at[index, 'ticker']
        current_date = asset_df.at[index, 'date']
        subset = asset_df[(asset_df['ticker'] == current_ticker) & (asset_df['date'] <= current_date)]
        variation =  (subset.high.iloc[-2] - subset.high.iloc[0] ) / len(subset)
        asset_df.at[index, 'high'] = subset.high.iloc[index-1] + variation
        asset_df.at[index, 'low'] = subset.high.iloc[index-1] - variation

    plt.figure(figsize=(10, 6))  # Taille facultative du graphique
    plt.plot(asset_df.date.tail(500), asset_df['high'].tail(500), label='High', color='blue')
    plt.plot(asset_df.date.tail(500), asset_df['low'].tail(500), label='Low', color='green')

    # Ajout de titres et légendes
    plt.title('High and Low Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Affichage du graphique
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calcul RSI -----------------------------------------------------------
    

    for index in range(asset_df[asset_df['close'].isna()].index[0], len(asset_df)):
        # Vérifier si nous avons une valeur dans 'close' pour la ligne précédente
        if index == 0 or pd.isna(asset_df.at[index-1, 'high']): # close
            continue  # Ignorer les premières lignes ou les lignes sans 'close' précédent

        # Calculer les variables explicatives pour la nouvelle ligne

        # Calcul de EMA (Exponential Moving Average)
        asset_df.at[index, 'ema'] = asset_df.loc[:index, 'close'].ewm(span=num_days).mean().iloc[-1]

        # Calcul de DEMA (Double Exponential Moving Average)
        ema = asset_df.loc[:index, 'close'].ewm(span=num_days).mean()
        asset_df.at[index, 'dema'] = 2 * ema.iloc[-1] - ema.ewm(span=num_days).mean().iloc[-1]

        # Calcul de Williams %R
        if index >= num_days:
            high = asset_df.loc[index-num_days:index, 'high'].max()
            low = asset_df.loc[index-num_days:index, 'low'].min()
            asset_df.at[index, 'williams'] = (high - asset_df.at[index, 'high']) / (high - low) * -100
        else:
            asset_df.at[index, 'williams'] = np.nan

        # Calcul de RSI (Relative Strength Index)
        delta = asset_df.loc[:index, 'close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=num_days).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=num_days).mean().iloc[-1]
        rs = gain / loss
        asset_df.at[index, 'rsi'] = 100 - (100 / (1 + rs))

        # Calcul de ADX (Average Directional Index)
        if index >= num_days:
            up_move = asset_df.loc[index-num_days:index, 'high'].diff(1)
            down_move = asset_df.loc[index-num_days:index, 'low'].diff(1).abs()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr = asset_df.loc[index-num_days:index, 'high'] - asset_df.loc[index-num_days:index, 'low']
            atr = tr.rolling(window=num_days).mean().iloc[-1]
            plus_di = 100 * (plus_dm / atr).mean()
            minus_di = 100 * (minus_dm / atr).mean()
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            asset_df.at[index, 'adx'] = dx
        else:
            asset_df.at[index, 'adx'] = np.nan

        # Calcul de standardDeviation (Écart-type)
        asset_df.at[index, 'standardDeviation'] = asset_df.loc[:index, 'close'].rolling(window=num_days).std().iloc[-1]

        # Calcul de Avg_close (Moyenne mobile de 'close')
        asset_df.at[index, 'Avg_close'] = asset_df.loc[:index, 'close'].rolling(window=num_days).mean().iloc[-1]




        # Suppression des lignes avec des valeurs NaN (créées par rolling et shift)
        asset_df = asset_df.dropna()
    return asset_df
    
def predict_new_close(dataframe, ticker: str): # TODO : check
    model = joblib.load(f'{ticker}_random_forest_model.pkl')

    new_X = dataframe[['volume', 'ema', 'dema', 'williams', 'rsi', 'adx', 'standardDeviation', 'Avg_close']]
    # ou plutot             year  month  day volume  ema  dema   williams   rsi  adx  standardDeviation   Avg_close
    # Faire les prédictions
    predictions = model.predict(new_X)

    # Ajouter les prédictions aux nouvelles données
    dataframe['close'] = dataframe['close'].fillna(predictions)

    return dataframe

def predict_new_close2(dataframe, ticker: str, days_pred:int = 10): # TODO : CHECK
    # Prédire les valeurs manquantes dans df_predict pour 'close'

    model = joblib.load(rf"CODE\ML_Models\random_forest_csv\models\{ticker}_random_forest_model.pkl")

    df_predict = dataframe[dataframe['close'].isna()]   
    histo_df = dataframe.head(len(dataframe) - days_pred)
    histo_df = histo_df[['year', 'month', 'day', 'volume', 'ema', 'dema', 'williams', 'rsi', 'adx', 'standardDeviation','Avg_close']]
    histo_df[f'ticker_{ticker}'] = True
    pred2 = model.predict(histo_df)
    df_predict['close'] = pred2[-days_pred:] 

    return df_predict

def main_method(asset_list, GridSearch:bool = False, plot:bool = False):
        
        confidence_level_df = pd.DataFrame(columns = ["ticker", "confidence_level"])
        final_cols = ['ticker', 'date','year','month','day', 'close','volume','ema','dema','williams','rsi','adx','standardDeviation','Avg_close']
        final_prevision_df = pd.DataFrame(columns = final_cols)
        df_pred_close_final = pd.DataFrame(columns = ['ticker', 'date', 'year', 'month', 'day', 'close'])

        for asset in asset_list:
            one_asset_df = format_df(dataset2, asset)

            if GridSearch : 
                param_grid = {
                    'n_estimators': [50 , 100, 1000],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [3,2,4] # on commence à 4 car racine de 12 vaut 3.4 i.e sqrt(n_features)
                }           ####### grid.loc[56].params <=== BEST PARAM 
                
                label, dataframe, features, features_list , dict_X_Y = ml_data_preparation(one_asset_df)

                # FIND BEST PARAMETERS : 
                grid_search_random_forest(train_features = dict_X_Y["train_features"] , 
                                            train_labels = dict_X_Y["train_labels"], 
                                            test_features = dict_X_Y["test_features"], 
                                            test_labels = dict_X_Y["test_labels"], 
                                            param_grid = param_grid, 
                                            path = r"CODE\ML_Models\random_forest_csv")
        
            best_params = {'max_depth': None, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}

            conf_level = create_train_model(asset_df = one_asset_df, asset_name = asset,  best_params = best_params) 
            data = {
                'ticker': [asset],
                'confidence_level': [conf_level]
            }
            df = pd.DataFrame(data)
            confidence_level_df = pd.concat([confidence_level_df, df], ignore_index=True)

            start_date = datetime.now() + timedelta(days=1)
            asset_df_new_dates = generate_dataframe(start_date)


            new_stock = pd.concat([one_asset_df, asset_df_new_dates], ignore_index=True)
            new_stock['date'] = pd.to_datetime(new_stock['date'])
            new_stock['ticker'] = new_stock.ticker[0]

            # stock_features_add = create_features(new_stock, stock)
            # df_histo_predic = predict_new_close(stock_features_add, stock)

            df_pred = predict_new_close2(new_stock, asset)
            df_pred_close = df_pred.copy()[['ticker', 'date', 'year', 'month', 'day', 'close']]
            df_pred_close_final = pd.concat([df_pred_close_final, df_pred_close], ignore_index=True)
            new_stock["close"] = new_stock["close"].fillna(df_pred.close)
            final_prevision_df = pd.concat([final_prevision_df, new_stock], ignore_index=True)
            if plot : 

                plt.plot(new_stock.date, new_stock['close'], label='Close')
                plt.title(f'Close Prices Over Time for {asset}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.show()
            
        return confidence_level_df, df_pred_close


if __name__ == "__main__":


    # Fetching the data
    print(">>> Getting data...")
    dataset2 = get_financial_data()
    print(">>> End Getting data! ")
    tickers_list = dataset2.ticker.unique().tolist()
    full_data_df = get_full_data_df(dataset2, tickers_list)
    niv_confiance_df, pred_df =  main_method(tickers_list, False)

    niv_confiance_df.to_csv(rf"CODE\ML_Models\random_forest_csv\previsions\confidence_level.csv")
    pred_df.to_csv(rf"CODE\ML_Models\random_forest_csv\previsions\prediction_df.csv")


    a = True