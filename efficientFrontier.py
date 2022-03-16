import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import scipy.optimize as sc

plt.style.use('seaborn-deep')
np.random.seed(537)

stockList = ['AAPL', 'AMZN', 'BAC', 'TSLA']
# stock = [stock + '.AX' for stock in stockList]
stock = stockList

endDate = dt.datetime.now()
startDate = endDate-dt.timedelta(days=365)

def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# print(getData(stock, startDate, endDate))

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(252)
    return returns, std

def randomPortfolios(numPortfolios, meanReturns, covMatrix, riskFreeRate):
    results = np.zeros((3, numPortfolios))
    weightsList = []
    for i in range(numPortfolios):
        weights = np.random.random(len(stock))
        weights /= np.sum(weights)
        weightsList.append(weights)
        portfolioReturn, portfolioStd = portfolioPerformance(weights, meanReturns, covMatrix)
        results[0,i] = portfolioStd
        results[1,i] = portfolioReturn
        results[2,i] = (portfolioReturn - riskFreeRate)/portfolioStd
    return results, weightsList

meanReturns, covMatrix = getData(stock, startDate, endDate)
numPortfolios = 20000
riskFreeRate = 0.05

def negativeSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns-riskFreeRate)/pStd

def maxSharpeRatio(meanReturns, covMatrix, riskFreeRate, constraintSet = (0,1)):
    "Minimize the negative Sharpe Ratio by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSharpeRatio, numAssets*[1./numAssets], args = args,
                         method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result


def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVariance(meanReturns, covMatrix, riskFreeRate, constraintSet = (0,1)):
    "Minimize the portfolio variance by changing the weights/allocation of assets in portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args = args,
                         method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result

def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]

def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    "For each return target, we want to optimize the portfolio for min variance"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args = args,
                         method = 'SLSQP', bounds = bounds, constraints = constraints)
    return effOpt


def efficientFrontier(meanReturns, covMatrix, returns_range):
    efficientList = []
    for ret in returns_range:
        efficientList.append(efficientOpt(meanReturns, covMatrix, ret))
    return efficientList

def display_calculated_ef_with_random(meanReturns, covMatrix, numPortfolios, riskFreeRate):
    results, _ = randomPortfolios(numPortfolios, meanReturns, covMatrix, riskFreeRate)
    
    maxSR = maxSharpeRatio(meanReturns, covMatrix, riskFreeRate)
    rp, sdp = portfolioPerformance(maxSR['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR.x,index=meanReturns.index,columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,2)for i in maxSR_allocation.allocation]
    maxSR_allocation = maxSR_allocation.T
    maxSR_allocation

    minVol = minimizeVariance(meanReturns, covMatrix, riskFreeRate)
    rp_min, sdp_min = portfolioPerformance(minVol['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol.x,index=meanReturns.index,columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,2)for i in minVol_allocation.allocation]
    minVol_allocation = minVol_allocation.T
    
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print (maxSR_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print (minVol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=5, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='o',color='r',s=100, label='Maximum Sharpe Ratio')
    plt.scatter(sdp_min,rp_min,marker='o',color='g',s=100, label='Minimum Volatility')

    target = np.linspace(rp_min, 0.32, 20)
    efficientPortfolios = efficientFrontier(meanReturns, covMatrix, target)
    plt.plot([p['fun'] for p in efficientPortfolios], target, linestyle='-', color='black', label='efficient frontier')
    plt.title('Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Returns')
    plt.legend(labelspacing=0.8)
    return plt.show()

display_calculated_ef_with_random(meanReturns, covMatrix, numPortfolios, riskFreeRate)