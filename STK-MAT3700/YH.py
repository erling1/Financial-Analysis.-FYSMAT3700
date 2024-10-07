
import yfinance as yf
import warnings
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt 


warnings.filterwarnings("ignore")


start = dt.datetime(2022,9,4)

end = dt.datetime.now()

df=yf.download("AAPL",start,end)

df.to_csv("APPLE.csv")

"""Exercise: download data of stock prices from some companies, and compute returns. 
 Choose daily and weekly data, annualise the estimated expectation and volatility for the returns."""


#using close prices for caculating returns

#calculating from 1 year back
start_2023 = dt.datetime(2023,9,13)

filtered_df = df.loc[start_2023:end]



days = [i for i in range(len(filtered_df['Close']))]

closing_prices = np.array(filtered_df['Close'])


#visulise Daily closing FTSE index prices from 1 year back
plt.plot(days,closing_prices)
plt.savefig('index prices')


#arr[1:] - arr[:-1]
daily_return = (closing_prices[1:] - closing_prices[:-1]) * 252
days_daily = np.array([i for i in range(len(daily_return) )])

plt.figure()  # Create a new figure
plt.plot(days_daily, daily_return)
plt.title('Daily Return')
plt.xlabel('Days')
plt.ylabel('Daily Return')
plt.savefig('daily_return.png')  # Save figure
plt.close()  # Close the figure to avoid overlap



#extracting only monday and friday closing prices
#issue is that there could be a varying amount of prices for each day
mondays_closing_price = np.array(filtered_df.loc[filtered_df.index.dayofweek == 0]['Close'])
fridays_closing_price = np.array(filtered_df.loc[filtered_df.index.dayofweek == 4]['Close'])

print(filtered_df)

weekly_return = (mondays_closing_price[1:] - fridays_closing_price[:-1]) * 52
days_weekly = np.array([i for i in range(len(weekly_return) )])




plt.figure()  # Create a new figure
plt.plot(days_weekly, weekly_return)
plt.title('Weekly Return')
plt.xlabel('Weeks')
plt.ylabel('Weekly Return')
plt.savefig('weekly_return.png')  # Save figure
plt.close()  # Close the figure to avoid overlap

"""
Volatility= srt( Variance )

Annualized_daily = daily_return * 252

Annualized_weekly = weekly_return * 52

Annualized Volatility=Daily Volatility× 252
​

Annualized Volatility=Weekly Volatility× 
52
​"""
