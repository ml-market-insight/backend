
from packages_import import *
from db_connection import *
import yfinance as yf
import fitz  # Import de PyMuPDF

# BOUNDERIES OF HISTORICAL DATA
start_date = '2017-01-01'
end_date = '2022-01-01'

# NUMBER OF TRADING DAY PER YEAR
NUM_OF_TRADING_DAYS = 252

# NUMBER OF PORTFOLIO TO GENERATE FOR MONTE-CARLO SIMULATIONS
NUM_PORTFOLIO = 10000


# _________________________________________________DOWNLOADING/DISPLAYING INITIAL DATA_________________________________




def show_data(data):
    # plotting a figure, precizing the size
    data.plot(figsize=(10, 5))

    # plotting using matplotlib
    plt.show()


# _________________________________________CALCULATING RETURNS/VOLATILITY_OF_ASSETS/PORTFOLIO___________________________


def calculate_return(data):
    # getting logarithmic return log(S(t+1)/S(t))
    log_return = np.log(data / data.shift(1))

    # First value is NaN
    return log_return[1:]


def show_statistics(returns):
    # Expected return per asset
    print(returns)
    print("Mean value of the returns (expected return per asset) :\n", returns.mean() * NUM_OF_TRADING_DAYS)

    # Covariance matrix of returns
    print("Covariance matrix of assets\n", returns.cov() * NUM_OF_TRADING_DAYS)


def show_mean_variance(returns, weights):
    # Annual return of the portfolio, based on asset returns, weighted by the allocation.
    portfolio_return = np.sum(returns.mean() * weights) * NUM_OF_TRADING_DAYS

    # Look at mathematical formula its simple to understand
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * NUM_OF_TRADING_DAYS, weights)))

    print("Expected portfolio return (mean) : ", portfolio_return)
    print("Expected portfolio volatility (standard deviation) : ", portfolio_volatility)


# _________________________________________________MONTE-CARLO_SIMULATIONS_____________________________________________


def show_portfolio(returns, volatility):
    # choosing dimension of the figure
    plt.figure(figsize=(10, 6))

    # plot volatility/returns, color depends on value
    plt.scatter(volatility, returns, c=returns / volatility, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected return')

    # Adding color bar sharpe ratio
    plt.colorbar(label='Sharpe ratio')
    plt.show()


def generate_portfolio(returns, stocks):
    # array of means (expected return of given portfolio)
    portfolio_means = []

    # array of volatility (risks of a given portfolio)
    portfolio_risks = []

    # array of weight (capital allocation in a given portfolio)
    portfolio_weight = []

    for v in range(NUM_PORTFOLIO):
        # creating random weight and normalizing them
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weight.append(w)

        # calculating the expected returns with the new weight
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_OF_TRADING_DAYS)

        # calculating the volatility related to the weight and returns of the portfolio
        portfolio_risks.append(np.sqrt(np.dot(w, np.dot(returns.cov() * NUM_OF_TRADING_DAYS, w))))

    # returning 3 numpy arrays
    return np.array(portfolio_weight), np.array(portfolio_means), np.array(portfolio_risks)


# _________________________________________________OPTIMIZING/FINDING_BEST_PORTFOLIO_____________________________________________


def statistics(weights, returns):
    # return the expected return of the portfolio
    portfolio_return = np.sum(returns.mean() * weights) * NUM_OF_TRADING_DAYS

    # return the expected volatility of the portfolio
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(returns.cov() * NUM_OF_TRADING_DAYS, weights)))

    # return an array contening the return, the volatility, the Sharpe Ratio
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


# the maximum sharpe ratio given by statistics is at a given f(x), so the minimum of -f(x)
def min_function_sharpe(weights, returns):
    # return the sharpe ratio
    return -statistics(weights, returns)[2]


# what are the constraints ? Sum of w =1
# f(x)=0 has to be minimized
def optimize_portfolio_vol(weights, returns, target_volatility):
    # Fonction de calcul de la volatilité
    def portfolio_volatility(weights, returns):
        return np.sqrt(np.dot(weights, np.dot(returns.cov() * NUM_OF_TRADING_DAYS, weights)))
    
    # Contraintes
    def portfolio_constraint(weights):
        return np.sum(weights) - 1.0  # La somme des poids doit être égale à 1
    
    def volatility_constraint(weights, returns, target_volatility):
        return portfolio_volatility(weights, returns) - target_volatility  # Contrainte pour la volatilité cible
    
    constraints = [
        {'type': 'eq', 'fun': portfolio_constraint},  # Contrainte de somme des poids
        {'type': 'eq', 'fun': lambda x: volatility_constraint(x, returns, target_volatility)}  # Contrainte pour la volatilité cible
    ]
    
    # Bornes pour les poids des actifs
    bounds = tuple((0, 1) for _ in range(len(weights)))
    
    # Utilisation de weights comme poids initiaux sous forme de tableau unidimensionnel
    x0 = np.array(weights)  # Assurez-vous que weights est un tableau unidimensionnel
    
    # Minimisation de la volatilité pour atteindre la volatilité cible
    result = optimization.minimize(fun=lambda x: portfolio_volatility(x, returns),  # Minimiser la volatilité
                                   x0=x0,
                                   method='SLSQP',
                                   bounds=bounds,
                                   constraints=constraints)
    
    return result

def optimize_portfolio(weights, returns, stocks):

    # The constraints is an equation
    # The function lambda takes x as argument
    # Check if the sum of x = 1 (sum of weights)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # define a tuple (0,1) for each assets in the portfolio
    bounds = tuple((0, 1) for _ in range(len(stocks)))

    # We use the min_function_sharpe function defined above
    # portfolio at index [0] is taken
    # passing the associated returns as argument
    # using SLSQP optimization method
    # The bounds for the weights of optimized portfolio is between 0 and 1
    # The sum of all weights is 1 as a constraint
    return optimization.minimize(fun=min_function_sharpe, x0=weights, args=returns, method='SLSQP', bounds=bounds,
                                 constraints=constraints)


def print_optimal_portfolio(_optimum, returns, page):
    # Optimum['x'] refers to the result of the optimization process
    opt_portfolio = _optimum['x'].round(3)
    print("Optimal portfolio: ", opt_portfolio)
    page.insert_text((50, 240), f"Target portfolio  : {opt_portfolio}")
    stat = statistics(opt_portfolio, returns)
    print("Expected return and sharpe ratio : ",stat)
    return page, opt_portfolio, stat


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols, page):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=6.4)
    plt.show()

