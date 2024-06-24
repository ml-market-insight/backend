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
print(">>> Import packages done !")


def get_full_stock_df(dataset):
    """
    Dataset with each stock information
    Return a dataframe containing each stock years of data
    """
    print(">>> get_full_stock_df() ... ")
    tickers = [stock for stock in dataset["ticker"] ]
    dataframe = pd.DataFrame( columns = ['date', 'close', 'volume', 'ema', 'dema', 'williams', 'rsi', 'adx', 'standardDeviation'])
    final_df = dataframe.copy()
    for stock in dataset['ticker'] : 
        dataset_one_stock = dataset[dataset["ticker"].isin([stock])].reset_index(drop=True)
        one_stock_df = pd.DataFrame(dataset_one_stock.loc[0, "time_series_data"])
        one_stock_df["ticker"] = dataset_one_stock.loc[0,"ticker"]
        final_df = pd.concat([final_df, one_stock_df])
    final_df = final_df[::-1].reset_index(drop = True)
    return final_df


def get_one_stock_df(dataset, stock):
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



def reformat_df(dataframe):
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



def dataframe_analysis_1(dataframe):

    df = dataframe[[ col for col in dataframe.columns.tolist() if col not in [ "date" , "year" , "month" , "day"] ]]

    print(f"Dataframe.head():\n", dataframe.head())
    print(f"Dataframe.info():\n", dataframe.info())
    print(f"Dataframe.describe():\n", df.describe())
    print(f"Dataframe.isnull().sum():\n", dataframe.isnull().sum())



def dataframe_analysis_2(dataframe, variable:str):
    """
    variable : une seul colonne à analyser
    """

    plt.figure(figsize=(12, 6))

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=dataframe[variable])
    plt.title(f"{variable.upper()} Boxplot")

    # Histogramme
    plt.subplot(1, 2, 2)
    sns.histplot(dataframe[variable], kde=True)
    plt.title(f"{variable.upper()} Histogramme")

    plt.tight_layout()
    plt.show()

    # Tracé de base (plot)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=dataframe, x=dataframe.index, y=variable) 
    plt.title(f"{variable.upper()} Tracé de base")
    plt.show()



def matrice_correlation(dataframe) : 
    df = dataframe[[col for col in dataframe.columns if col not in ['ticker']]]
    plt.figure(figsize=(15,10))
    matrice = df.corr()
    sns.heatmap(matrice,annot=True)
    plt.title('Correlation')
    plt.show()

def avg_close_price(dataframe):
    df = dataframe.copy()
    df['Avg_close'] = df['close'].rolling(window=5, min_periods=1).mean()
    return df

def outliers_handle(dataframe):
    pass


def ml_data_preparation(dataframe):

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
    
    return labels, dataframe, features, features_list , train_test_dict
    



def ml_model_baseline(dict, list_features):
    test_features = dict_X_Y["test_features"]
    baseline_preds = test_features[:, list_features.index('Avg_close')]
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - dict_X_Y["test_labels"])
    avg_baseline_error = round(np.mean(baseline_errors), 2)
    print('Average baseline error in degree : ', avg_baseline_error )
    return avg_baseline_error


def RFR_model_1(dict):
        # Train the model on the training data : 
        # Instancier le modèle
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        # Train du modèle
    rf.fit(dict["train_features"], dict["train_labels"])
        # Use the forest's predict method on the test data
    predictions = rf.predict(dict["test_features"])
        # Calculate the absolute errors
    errors = abs(predictions - dict["test_labels"])
        # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / dict["test_labels"])
        # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    r2 = r2_score(dict['test_labels'], predictions)
    mse = mean_squared_error(dict['test_labels'], predictions)
    rmse = np.sqrt(mse)
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")


def RFR_model_2(dict):
    X_train = dict['train_features']
    y_train = dict['train_labels']
    X_test = dict['test_features']
    y_test = dict['test_labels']

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)


def grid_search_random_forest(train_features, train_labels, test_features, test_labels, param_grid, path):
    
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



if __name__ == "__main__":


    # Fetching the data
    stocks = ['AAPL']
    print(">>> Getting data...")
    dataset2 = get_financial_data()
    print(">>> End Getting data! ")

    full_stock_df = get_full_stock_df(dataset2)
    one_stock_df = get_one_stock_df(dataset2, ["AAPL"])
    one_stock_df = reformat_df(one_stock_df)


    one_stock_df = avg_close_price(one_stock_df)
    """
    dataframe_analysis_1(one_stock_df)
    dataframe_analysis_2(one_stock_df, "rsi")
    matrice_correlation(one_stock_df)
    """

    
    """
    State the question and determine required data : Prédire l'évolution de la variable Close
    Acquire the data in an accessible format : C'est fait, il manque aucune data dans one_stock pour AAPL et il en manque 19 sur tous les indicateurs dans full_stock_df
    Identify and correct missing data points/anomalies as required : Aucune dans one_stock mais faire un code pour trouver et gérer
    """
    # Prepare the data for the machine learning model :
    
    outliers_handle(one_stock_df) # TODO : faire cette fonction
    one_stock_df = one_stock_df[[col for col in one_stock_df.columns.tolist() if col != "date"]]
    label, dataframe, features, features_list , dict_X_Y = ml_data_preparation(one_stock_df)
    print('Training Features Shape:', dict_X_Y["train_features"].shape) # on a bien les données anciennes dans le train 
    print('Training Labels Shape:', dict_X_Y["train_labels"].shape) # on a bien les données anciennes dans le train
    print('Testing Features Shape:', dict_X_Y["test_features"].shape) # on a bien les données nouvelles dans le test
    print('Testing Labels Shape:', dict_X_Y["test_labels"].shape) # on a bien les données nouvelles dans le test


    # Establish a baseline model that you aim to exceed :
    avg_baseline_error = ml_model_baseline(dict_X_Y, features_list)
    
    RFR_model_1(dict_X_Y)

    # GRID pour GridSearchCV
    param_grid2 = {
        'n_estimators': [50 , 100, 250, 500, 750, 1000],
        'max_depth': [None, 10, 20, 30,50,100],
        'min_samples_split': [2, 5, 10, 35, 50 , 75, 100],
        'min_samples_leaf': [1, 2, 4 , 10 , 20 , 30],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    param_grid = {
        'n_estimators': [50 , 100, 1000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [3,2,4] # on commence à 4 car racine de 12 vaut 3.4 i.e sqrt(n_features)
        # avec ce param_grid la fonction grid_search_random_forest run de 5h30 à 9h30 : 4h30
    } ####### grid.loc[56].params <=== BEST PARAM 
    
    #print("GRID SEARCH RANDOM FOREST")
    """grid_search_random_forest(train_features = dict_X_Y["train_features"] , 
                              train_labels = dict_X_Y["train_labels"], 
                              test_features = dict_X_Y["test_features"], 
                              test_labels = dict_X_Y["test_labels"], 
                              param_grid = param_grid, 
                              path = r"CODE\ML_Models\random_forest_csv")"""

    """
    n_estimators : Nombre d'arbres dans la forêt : Typiquement entre 100 et 1000
    max_depth : Profondeur maximale des arbres. Peut être None (pour une profondeur illimitée) ou des valeurs comme 10, 20, 30.
    min_samples_split : Nombre minimum d'échantillons requis pour diviser un nœud interne. Entre 2 et 10.
    min_samples_leaf : Nombre minimum d'échantillons requis pour être à un nœud terminal (feuille). Entre 1 et 10
    max_features : Nombre de caractéristiques à considérer lors de la recherche de la meilleure division.  Peut être une fraction comme 0.5, sqrt ou log2
    """


    a = True