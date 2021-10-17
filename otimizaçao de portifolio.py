#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:42:16 2021

@author: lucasbabrikowski
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf  
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

today = datetime.today().strftime('%Y-%m-%d')
assets = [i for i in input("escreva as ações separadas por vírgula: ").split(",")]
tipo = input("escolha: 1 - Max sharpe 2 - Min vol: ")
min_w = input("% mínima por ativo escolhido: ")
max_w = input("% máxima por ativo escolhido: ")
TLR = 0.02


len_assets = len(assets)
w1= 1/len_assets
weights = [w1]*len_assets
print(weights)
# Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

# Store the adjusted close price of the sock into the df
for stock in assets:
  df1 = yf.download(assets,'2020-01-01')
  
df = df1['Adj Close']


returns = df.pct_change()
cov_matrix_annual = returns.cov()*252
port_variance = np.dot(weights,np.dot(cov_matrix_annual,weights))
port_volatility = np.sqrt(port_variance)
portfolio_simple_annual_return = np.sum(returns.mean()*weights)*252

# Sow the expected annual return, volatility (risk), and variance
percent_var = str(round(port_variance,2)*100)+'%'
percent_vola = str(round(port_volatility,2)*100)+'%'
percent_ret = str(round(portfolio_simple_annual_return,2)*100)+'%'

print("Dados Originais:")
print('Retorno anual esperado: '+percent_ret)
print('Risco/Vol anual: '+percent_vola)
print('Variancia anual: '+percent_var)
print("----------------------------------------------------")


#if(tipo=="1"):
# Portfolio Optimization

# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)


#for w in range(0,len_assets):
   # w = "w"+str([w])+"+"

if tipo == "1":
    # Optimize for maximum sharpe ratio
    ef = EfficientFrontier(mu,S,weight_bounds=(min_w,max_w))
    #ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3] == 1)
    weights = ef.max_sharpe(risk_free_rate=TLR)
    cleaned_weights = ef.clean_weights() 
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)
    
    # Get the discrete allocation of each share per stock
    import cvxpy
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    
    latest_prices = get_latest_prices(df)
    weights = cleaned_weights
    da = DiscreteAllocation(weights,latest_prices,total_portfolio_value = 6000)
    allocation,leftover = da.lp_portfolio()
    print("Dados Otimizados:")
    print('Discrete allocation: ',allocation)
    print('Funds remaining: ${:.2f}'.format(leftover))
    
else:
       # Optimize for maximum sharpe ratio
    ef = EfficientFrontier(mu,S,weight_bounds=(min_w,max_w))
    #ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3] == 1)
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights() 
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)
    
    # Get the discrete allocation of each share per stock
    import cvxpy
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    
    latest_prices = get_latest_prices(df)
    weights = cleaned_weights
    da = DiscreteAllocation(weights,latest_prices,total_portfolio_value = 6000)
    allocation,leftover = da.lp_portfolio()
    print('Discrete allocation: ',allocation)
    print('Funds remaining: ${:.2f}'.format(leftover)) 

"""""

max_quadratic_utility() maximises the quadratic utility, given some risk aversion.
efficient_risk() maximises return for a given target risk

efficient_return(target_return, market_neutral=False)[source]
Calculate the ‘Markowitz portfolio’, minimising volatility for a given target return.

Parameters:	
target_return (float) – the desired return of the resulting portfolio.
market_neutral (bool, optional) – whether the portfolio should be market neutral (weights sum to zero), defaults to False. Requires negative lower weight bound.


efficient_return() minimises risk for a given target return
"""""

"""""
# Visually show the stock/ portfolio
title = 'Portfolio Adj. Close Price History'

# Get the stocks
my_stocks = df

# Create and plot the graph
for c in my_stocks.columns.values:
  plt.plot(my_stocks[c],label=c)

plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj. Price USD ($)',fontsize=18)
plt.legend(my_stocks.columns.values,loc='upper left')
plt.show()

print(assets)
print(weights)

# In[ ]:


# Portfolio Optimization

# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximum sharpe ratio
ef = EfficientFrontier(mu,S,weight_bounds=(None,None))
ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3] == 1)
plotting.plot_efficient_frontier(ef)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights() 
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[ ]:


# Portfolio Optimization

# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximum sharpe ratio
ef = EfficientFrontier(mu,S,weight_bounds=(None,None))
ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3] == 1)
# 100 portfolios with risks between 0.10 and 0.30
risk_range = np.linspace(0.30, 0.80, 1000)
plotting.plot_efficient_frontier(ef, ef_param="risk", ef_param_range=risk_range,show_assets=True, showfig=True)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights() 
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[ ]:


# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximum sharpe ratio
ef = EfficientFrontier(mu,S,weight_bounds=(None,None))
ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3] == 1)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt(np.diag(w @ S @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()
"""""



