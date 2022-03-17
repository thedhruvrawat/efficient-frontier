import argparse
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import scipy.optimize as sc

plt.style.use('seaborn')
np.random.seed(537)

cli=argparse.ArgumentParser()
cli.add_argument(
  "--stocks",  
  help='The stocks to include in portfolio',
  nargs="*",  
  type=str,
  default=['AAPL', 'AMZN', 'BAC', 'TSLA'],  
)
cli.add_argument(
  "--num",  
  help='The number of portfolios to be simulated',
  nargs=1, 
  type=int,
  default=25000, 
)
cli.add_argument(
  "--rfr",  
  help='The risk free rate of return',
  nargs=1, 
  type=float,
  default=0.075,  
)
cli.add_argument(
  "--years",  
  help='The number of years',
  nargs=1, 
  type=int,
  default=[1],  
)

args = cli.parse_args()

# stockList = ['AAPL', 'AMZN', 'BAC', 'TSLA']
# stock = [stock + '.AX' for stock in stockList]
stock = args.stocks
numPortfolios = args.num
riskFreeRate = args.rfr
yr = [str(integer) for integer in args.years]
y = "".join(yr)
yr = int(y)
day = yr*365
mktDay = yr*252

endDate = dt.datetime.now()
startDate = endDate-dt.timedelta(days=day)

def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    # print(stockData)
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix, returns, stockData

# print(getData(stock, startDate, endDate))

def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*(mktDay)
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(mktDay)
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

meanReturns, covMatrix, returns, table = getData(stock, startDate, endDate)


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
    # print(maxSR)
    rp, sdp = portfolioPerformance(maxSR['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR.x,index=meanReturns.index,columns=['Allocation'])
    maxSR_allocation.Allocation = [round(i*100,2)for i in maxSR_allocation.Allocation]
    maxSR_allocation = maxSR_allocation.T
    maxSR_allocation

    minVol = minimizeVariance(meanReturns, covMatrix, riskFreeRate)
    rp_min, sdp_min = portfolioPerformance(minVol['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol.x,index=meanReturns.index,columns=['Allocation'])
    minVol_allocation.Allocation = [round(i*100,2)for i in minVol_allocation.Allocation]
    minVol_allocation = minVol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(mktDay)
    an_rt = meanReturns * mktDay

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
    print ("Individual Stock Returns and Volatility\n")
    maxEF=0
    minEF=1
    maxVol=0
    for i, txt in enumerate(table.columns):
        if(an_rt[i]>maxEF):
            maxEF=an_rt[i]
        if(an_rt[i]<minEF):
            minEF=an_rt[i]
        if(an_vol[i]>maxVol):
            maxVol=an_vol[i]
        print(txt,":","Annualized Return = ", round(an_rt[i]*100,2), ", Annualized Volatility = ",round(an_vol[i]*100,2))
    print ("-"*80)
    
    plt.subplots(figsize=(10, 7))
    # plt.margins(y=0)
    plt.scatter(an_vol,an_rt,marker='o', s=20, color='black')
    
    x = np.linspace(riskFreeRate, maxEF, 50)
    cml = riskFreeRate-x*maxSR['fun'] # Since negative of Sharpe Ratio is being maximized
    plt.plot(x, cml,'b', label="Capital Market Line")
    # plt.plot([0,riskFreeRate],[sdp_min,rp], 'b', label="Capital Market")

    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='plasma', marker='o', s=5, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    # plt.scatter(sdp,rp,marker='o',color='r',s=50, label='Maximum Sharpe Ratio')
    # plt.scatter(sdp_min,rp_min,marker='o',color='g',s=50, label='Minimum Volatility')
    for i, txt in enumerate(table.columns):
        plt.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    plt.scatter(sdp,rp,marker='D',color='r',s=50, label='Maximum Sharpe Ratio')
    plt.scatter(sdp_min,rp_min,marker='D',color='g',s=50, label='Minimum Volatility')

    target = np.linspace(rp_min, maxEF, 50)
    efficientPortfolios = efficientFrontier(meanReturns, covMatrix, target)
    plt.plot([p['fun'] for p in efficientPortfolios], target, 'k--', label='Efficient Frontier')
    plt.title('Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Returns (%)')
    plt.legend(labelspacing=0.8)
    # plt.ylim(0.5*minEF,1.15*maxEF)
    # plt.xlim(0.85*sdp_min,1.05*maxVol)
    # plt.plot([p['fun'] for p in efficientPortfolios], target, 'r--', label='Efficient Frontier')
    # plt.set_title('Portfolio Optimization with Individual Stocks')
    # plt.set_xlabel('Annualized Volatility (%)')
    # plt.set_ylabel('Annualised Returns (%)')
    # plt.legend(labelspacing=0.8)
    return plt.show()

def main():
    print("Stocks: ", stock)
    print("Risk Free Rate: ", riskFreeRate)
    display_calculated_ef_with_random(meanReturns, covMatrix, numPortfolios, riskFreeRate)

if __name__ == "__main__":
    main()
