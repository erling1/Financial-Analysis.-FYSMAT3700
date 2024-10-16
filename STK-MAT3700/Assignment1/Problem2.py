import math
import yfinance as yf
import warnings
import datetime as dt
import requests
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

start = dt.datetime(2023,10,1)
end = dt.datetime.now()

#Retriving data from five different companies
df_Apple = yf.download("AAPL",start,end)

print(df_Apple.columns)

class BlackAndScholes:

    def __init__(self, S_0: float, K: float, T: int, r: float, sigma: int, option='call'):
        """
    A class to calculate the price of European options using the Black-Scholes model.

    Attributes:
    -----------
    S0 : float
        The spot price of the underlying asset.
    K : float
        The strike price of the option.
    T : int
        The time to expiration in years.
    r : float
        The risk-free interest rate.
    sigma : int
        The volatility of the underlying asset.
    option : str, optional
        The type of option, either 'call' or 'put' (default is 'call').
    """
        self.S = S_0            
        self.K = K            
        self.T = T            
        self.r = r            
        self.sigma = sigma    
        self.option = option  
        

    def black_and_scholes(self):
        """
        Initializes the Black-Scholes model with the given parameters.
        """
        S = self.S             
        K = self.K              
        T = self.T              
        r = self.r              
        sigma = self.sigma       
        option = self.option     

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)  

        if option == 'call':
            return S * math.exp(-r * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif option == 'put':
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-r * T) * norm.cdf(-d1)




S0 = df_Apple["Close"][-1]
K_plus =  (S0, S0 + 0.1 * S0, S0 + 0.3 * S0)
K_minus =  (S0, S0 - 0.1 * S0, S0 - 0.3 * S0)
volatilities = [0.10, 0.3, 0.45]
T = (1/12, 3/12, 6/12)

#A bit unsure if these are at all correct rates, might have misunderstood
df = pd.read_csv("daily-treasury-rates.csv")

r = (df["1 Mo"].mean(), df["3 Mo"].mean(), df["6 Mo"].mean())

strike_prices_plus = list(K_plus)
strike_prices_minus = list(K_minus)

for sigma in volatilities:
    #call_prices = []
    for i in T:
        call_prices_plus = []
        call_prices_minus = []
        for j,k in zip(K_plus,K_minus):
            call_plus = BlackAndScholes(S0, j, sigma, i, r[0] / 100)
            call_minus = BlackAndScholes(S0, k, sigma, i, r[0] / 100)
        
            call_prices_plus.append(call_plus.black_and_scholes())
            call_prices_minus.append(call_minus.black_and_scholes())
    
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(strike_prices_plus, call_prices_plus, marker='o', color = "orange",label=f'K plus S0')
        plt.plot(strike_prices_minus, call_prices_minus, marker='o',color = "blue", label=f'K minus S0 ')

        # Adding titles and labels
        plt.title(f"Call Option Prices for Volatility: {sigma:.2f}, Time: {i:.2f}", fontsize=16)
        plt.xlabel('Strike Price', fontsize=14)
        plt.ylabel('Call Option Price', fontsize=14)

        """# Customizing ticks
        plt.xticks(strike_prices_plus)  # Set x-ticks to the strike prices
        plt.yticks(np.arange(0, max(call_prices_plus)+5, 5))  # Customize y-ticks"""

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)  # Add a grid with some transparency

        # Add legend
        plt.legend()

        # Save the plot to a file
        plt.tight_layout()  # Adjust the layout to fit everything nicely
        plt.savefig(f"BlackScholes_Vol_{sigma:.2f}_Time_{i:.2f}.png")

#b)

stock = yf.Ticker("AAPL")

exp_date = stock.options[5]
options_chain = stock.option_chain(exp_date)

call_options = options_chain.calls


current_price = S0
filtered_df_HigherPrice = call_options[call_options["strike"] > current_price].iloc[-2:]  # Options with strike above current price
filtered_df_LowerPrice = call_options[call_options["strike"] < current_price].iloc[:2]

tolerance = 5
filtered_df_sameprice = call_options[(call_options["strike"] >= current_price - tolerance) &(call_options["strike"] <= current_price + tolerance)]

print(filtered_df_sameprice)


print(len(filtered_df_HigherPrice))
print(len(filtered_df_LowerPrice))

def imp_vol(market_price):
    S_0 = S0
    K =  S_0
    T = 1/12
    r = 5

    
    #solving for volitiity
    def func_to_optimize(sigma_0):
        Black_Scholes = BlackAndScholes(S_0, K, T, r, sigma_0)
        #f(x) = 0
        return Black_Scholes.black_and_scholes() - market_price
    

    # Use fsolve to find the sigma that minimizes the difference
    initial_guess = 2  # Initial guess for volatility (20%)
    implied_vol = fsolve(func_to_optimize, initial_guess, xtol= 0.005)[0]
    
    return implied_vol



combined_df = pd.concat([filtered_df_HigherPrice, filtered_df_LowerPrice, filtered_df_sameprice])
combined_df['imp_vols'] = np.nan


print(combined_df)
strikes = combined_df["strike"]

for index, row in combined_df.iterrows():
    implied_vol = imp_vol(row['strike'])  
    combined_df.at[index, 'imp_vols'] = implied_vol

print(combined_df[["imp_vols","strike"]])

# Plotting
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(combined_df["strike"], combined_df["imp_vols"], marker='o', color='b', linestyle='-', label='Implied Volatility')


plt.title('Implied Volatility vs Strike Price', fontsize=16)
plt.xlabel('Strike Price', fontsize=14)
plt.ylabel('Implied Volatility', fontsize=14)


plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc='best', fontsize=12)


plt.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('implied_volatility_vs_strike.png', dpi=300)