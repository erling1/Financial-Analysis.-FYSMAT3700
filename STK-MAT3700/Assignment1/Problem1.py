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

#Retriving data from five different companies
df_Apple = yf.download("AAPL",start,end)
df_Nvidia = yf.download("NVDA",start,end)
df_JPM = yf.download("JPM",start,end)
df_Goldman = yf.download("GS",start,end)
df_Exxon = yf.download("XOM",start,end)


df_Apple.to_csv("APPLE.csv")
df_Nvidia.to_csv("NVDA")
df_JPM.to_csv("JPM")
df_Goldman.to_csv("GS")
df_Exxon.to_csv("XOM")

dataframes = {"apple" : df_Apple.loc[start:end], 
              "nvidia" : df_Nvidia.loc[start:end], 
              "jpm" : df_JPM.loc[start:end], 
              "goldman" : df_Goldman.loc[start:end], 
              "exxon" : df_Exxon.loc[start:end], }



#Checking dataframe features
print(dataframes["nvidia"].columns)

#Adding all closing prices to a prices dictionary
Closing_prices =  { "apple" : np.array(dataframes["apple"]["Close"]), 
                    "nvidia" :  np.array(dataframes["nvidia"]["Close"]), 
                    "jpm" :  np.array(dataframes["jpm"]["Close"]), 
                    "goldman" :  np.array(dataframes["goldman"]["Close"]), 
                    "exxon" :  np.array(dataframes["exxon"]["Close"] )}


daily_returns = {   "apple": (Closing_prices["apple"][1:] - Closing_prices["apple"][:-1]) / Closing_prices["apple"][:-1],
                    "nvidia": (Closing_prices["nvidia"][1:] - Closing_prices["nvidia"][:-1]) / Closing_prices["nvidia"][:-1],
                    "jpm": (Closing_prices["jpm"][1:] - Closing_prices["jpm"][:-1]) / Closing_prices["jpm"][:-1],
                    "goldman": (Closing_prices["goldman"][1:] - Closing_prices["goldman"][:-1]) / Closing_prices["goldman"][:-1],
                    "exxon": (Closing_prices["exxon"][1:] - Closing_prices["exxon"][:-1]) / Closing_prices["exxon"][:-1]
                }




annualized_returns = {key : returns.mean()*252 for key, returns in daily_returns.items()}

volatility = {key : returns.std() for key, returns in daily_returns.items()}

annualized_volatility = {key: vol * np.sqrt(252) for key, vol in volatility.items()}

number_of_companies = 5

probabilities = [] 

for key,item in dataframes.items():
    returns = pd.Series(daily_returns[key])

    #Create a range of values (for plotting the fitted normal distribution)
    x = np.linspace(returns.min(), returns.max(), 100)

    x_probabilites = np.linspace(returns.min(), returns.max(), len(daily_returns[key]))

    plt.figure()
    plt.hist(daily_returns[key], bins=30, density=True, alpha=0.5, color='blue', label='Empirical Density')

    mean_return = daily_returns[key].mean()
    std_return = daily_returns[key].std()

    probabilities.append(norm.pdf(x_probabilites, mean_return, std_return))

    # Plot the fitted normal density
    plt.title(f"{key}")
    plt.plot(x, norm.pdf(x, mean_return, std_return), color='red', label='Fitted Normal Density')
    plt.xlabel('Closing Price')
    plt.ylabel('Density') #likilihood
    plt.savefig(f"Task1_empirical_and fitted_density_{key}")
   
    

#b)
print("Task b)\n")
    
#Turning daily_returns into a dataframe for the pandas corr() method
    
corr_dataframe = pd.DataFrame(daily_returns)
correlation_matrix = corr_dataframe.corr()

plt.figure()
plt.title('Correlation Heatmap')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
            vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
plt.savefig("Correlation between all five assets")

Covariance_matrix = volatility * correlation_matrix * volatility
Covariance_matrix.reset_index(drop=True, inplace=True)

# Remove the column names
Covariance_matrix.columns = [''] * len(Covariance_matrix.columns)

print("\n")
print("Covariance matrix:")
print(Covariance_matrix)


Expected_returns =  np.array([value.mean() for value in daily_returns.values()])

number_of_portfolios = 1000



portofolio_returns = np.zeros(number_of_portfolios)
portofolio_risks = np.zeros(number_of_portfolios)

def portfolio_variance(weights, covariance_matrix):
    return weights.T @ covariance_matrix @ weights



for i in range(number_of_portfolios):

    weights = np.random.rand(5)
    weights /= np.sum(weights)

    portofolio_risk = np.sqrt(portfolio_variance(weights,Covariance_matrix,))

    portofolio_return = weights @ Expected_returns

    portofolio_returns[i] = portofolio_return
    portofolio_risks[i] = portofolio_risk


constraints = ({'type': 'eq', 
                'fun': lambda weights: np.sum(weights) - 1})

#Ensure weights are between 0 and 1 
bounds = tuple((0, 1) for asset in range(5))

weigths0 = np.ones(5) / 5

optimal_variance = minimize(portfolio_variance, weigths0, args= (Covariance_matrix,),method='SLSQP',bounds=bounds, constraints=constraints)

optimal_weigths = optimal_variance.x

minimum_variance_portfolio_return = optimal_weigths @ Expected_returns
minimum_variance_portofolio_risk = np.sqrt(portfolio_variance(optimal_weigths,Covariance_matrix,))

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






    
