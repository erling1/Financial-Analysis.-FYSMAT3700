import yfinance as yf
import warnings
import datetime as dt
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize


#Defining Timeframe
start = dt.datetime(2023,10,1)
end = dt.datetime.now()


class EfficentFronter:
    def __init__(self, timeframe: tuple, tickers: list) -> None:
        self.start = timeframe[0]
        self.end = timeframe[1]
        self.tickers = tickers
        self.ticker_dataframes = {}
        self.closing_prices_dict = {}
        self.daily_returns_dict = {}
        self.annualized_returns = {}
        self.volatility_dict = {}
        self.annualized_volatility_dict = {}

        # Automatically perform calculations
        self.download_data()
        self.closing_prices()
        self.daily_returns()
        self.calculate_annualized_return()
        self.calculate_volatility()
        self.calculate_annualized_volatility()

    def download_data(self):
        # Download data for each ticker
        for ticker in self.tickers:
            self.ticker_dataframes[ticker] = yf.download(ticker, self.start, self.end).loc[self.start:self.end]

    def closing_prices(self):
        # Calculate closing prices for each ticker
        for ticker in self.tickers:
            self.closing_prices_dict[ticker] = np.array(self.ticker_dataframes[ticker]["Close"])

    def daily_returns(self):
        # Calculate daily returns for each ticker
        for ticker in self.tickers:
            self.daily_returns_dict[ticker] = (self.closing_prices_dict[ticker][1:] - self.closing_prices_dict[ticker][:-1]) / self.closing_prices_dict[ticker][:-1]

    def calculate_annualized_return(self):
        # Calculate annualized return for each ticker
        self.annualized_returns = {key: returns.mean() * 252 for key, returns in self.daily_returns_dict.items()}

    def calculate_volatility(self):
        # Calculate daily volatility (std dev) for each ticker
        self.volatility_dict = {key: returns.std() for key, returns in self.daily_returns_dict.items()}

    def calculate_annualized_volatility(self):
        # Calculate annualized volatility for each ticker
        self.annualized_volatility_dict = {key: vol * np.sqrt(252) for key, vol in self.volatility_dict.items()}

    def portfolio_variance(self,weights, covariance_matrix):

        return weights.T @ covariance_matrix @ weights
    

    def plot_efficent_frontier(self, number_of_portfolios):

        corr_dataframe = pd.DataFrame(self.daily_returns_dict)
        correlation_matrix = corr_dataframe.corr()

        Covariance_matrix = self.volatility_dict * correlation_matrix * self.volatility_dict

        Expected_returns =  np.array([value.mean() for value in self.volatility_dict.values()])

        
        portofolio_returns = np.zeros(number_of_portfolios)
        portofolio_risks = np.zeros(number_of_portfolios)



        for i in range(number_of_portfolios):

            weights = np.random.rand(len(self.tickers))
            weights /= np.sum(weights)

            portofolio_risk = np.sqrt(self.portfolio_variance(weights,Covariance_matrix,))

            portofolio_return = weights @ Expected_returns

            portofolio_returns[i] = portofolio_return
            portofolio_risks[i] = portofolio_risk


        constraints = ({'type': 'eq', 
                        'fun': lambda weights: np.sum(weights) - 1})

        #Ensure weights are between 0 and 1 
        bounds = tuple((0, 1) for asset in range(len(self.tickers)))

        weigths0 = np.ones(len(self.tickers)) / len(self.tickers)

        optimal_variance = minimize(self.portfolio_variance, weigths0, args= (Covariance_matrix,),method='SLSQP',bounds=bounds, constraints=constraints)

        optimal_weigths = optimal_variance.x

        minimum_variance_portfolio_return = optimal_weigths @ Expected_returns
        minimum_variance_portofolio_risk = np.sqrt(self.portfolio_variance(optimal_weigths,Covariance_matrix,))

        plt.figure(figsize=(10, 6))
        plt.scatter(portofolio_risks, portofolio_returns, c=portofolio_returns, cmap='viridis', marker='o', alpha=0.3)
        plt.scatter(minimum_variance_portofolio_risk, minimum_variance_portfolio_return, color='red',marker='x',s=200, linewidths=3, alpha=1, label='Minimum Variance Portfolio'
        )
        plt.colorbar()
        plt.title('Efficient Frontier')
        plt.xlabel('Portfolio Risk ')
        plt.ylabel('Portfolio Return')
        plt.grid(True)
        plt.savefig("Efficent Frontier")
        plt.show()



list_of_tickers = ["AAPL", "NVDA"]
timeframe = (start,end)
test = EfficentFronter(timeframe, list_of_tickers)
test.plot_efficent_frontier(1000)
   
    


