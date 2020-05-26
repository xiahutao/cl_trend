import datetime
import numpy as np
import math
import logging


class VolatilityPricer():
    """
    Realized vol:
    Same as Black-Scholes, we assume the underlying follows a Geometric Brownian Motion.
    Then its log return follows a Normal distribution, with mean as 0.
    We take as input the historical daily underlying prices.
    Annualization factor is 365.
    Degree of Freedom is 0 as we are calculating the exact realized vol for the given historical period.

    Implied vol:
    Use Black-Scholes to back out the implied volatility from the given market option price.

    """

    def __init__(self, df):
        self.historicalDataBySymbol = dict()
        self.realizedVolBySymbol = dict()
        self.close = df.colse

    def _calculateRealizedVol(self, ts):
        """ Calculate the realized vol from given time series """
        pctChange = ts.pct_change().dropna()
        logReturns = np.log(1 + pctChange)
        vol = np.sqrt(np.sum(np.square(logReturns)) / logReturns.size)
        annualizedVol = vol * np.sqrt(365)
        return annualizedVol

    def getRealizedVol(self):
        """ Calculate the realized volatility from historical market data """
        realizedVol = self._calculateRealizedVol(self.close)
        self.realizedVolBySymbol = realizedVol
        return self.realizedVolBySymbol

    def getImpliedVol(self, optionPrice=17.5, callPut='Call', spot=586.08, strike=585.0, tenor=0.109589, rate=0.0002):
        """ Calculate the implied volatility from option market price """
        return self.blackScholesSolveImpliedVol(optionPrice, callPut, spot, strike, tenor, rate)

    def blackScholesOptionPrice(self, callPut, spot, strike, tenor, rate, sigma):
        """
        Black-Scholes option pricing
        tenor is float in years. e.g. tenor for 6 month is 0.5
        """
        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
        d2 = d1 - sigma * np.sqrt(tenor)
        if callPut == 'Call':
            return spot * ncdf(d1) - strike * np.exp(-rate * tenor) * ncdf(d2)
        elif callPut == 'Put':
            return -spot * ncdf(-d1) + strike * np.exp(-rate * tenor) * ncdf(-d2)

    def blackScholesVega(self, callPut, spot, strike, tenor, rate, sigma):
        """ Black-Scholes vega """
        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
        return spot * np.sqrt(tenor) * npdf(d1)

    def blackScholesDelta(self, callPut, spot, strike, tenor, rate, sigma):
        """ Black-Scholes delta """
        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
        if callPut == 'Call':
            return ncdf(d1)
        elif callPut == 'Put':
            return ncdf(d1) - 1

    def blackScholesGamma(self, callPut, spot, strike, tenor, rate, sigma):
        """" Black-Scholes gamma """
        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tenor) / (sigma * np.sqrt(tenor))
        return npdf(d1) / (spot * sigma * np.sqrt(tenor))

    def blackScholesSolveImpliedVol(self, targetPrice, callPut, spot, strike, tenor, rate):
        """" Solve for implied volatility using Black-Scholes """
        MAX_ITERATIONS = 100
        PRECISION = 1.0e-5

        sigma = 0.5
        i = 0
        while i < MAX_ITERATIONS:
            optionPrice = self.blackScholesOptionPrice(callPut, spot, strike, tenor, rate, sigma)
            vega = self.blackScholesVega(callPut, spot, strike, tenor, rate, sigma)
            diff = targetPrice - optionPrice
            logging.debug('blackScholesSolveImpliedVol: iteration={}, sigma={}, diff={}'.format(i, sigma, diff))
            if abs(diff) < PRECISION:
                return sigma
            sigma = sigma + diff / vega
            i = i + 1
        logging.debug(
            'blackScholesSolveImpliedVol: After MAX_ITERATIONS={}, best sigma={}'.format(MAX_ITERATIONS, sigma))
        return sigma


def ncdf(x):
    """
    Cumulative distribution function for the standard normal distribution.
    Alternatively, we can use below:
    from scipy.stats import norm
    norm.cdf(x)
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def npdf(x):
    """
    Probability distribution function for the standard normal distribution.
    Alternatively, we can use below:
    from scipy.stats import norm
    norm.pdf(x)
    """
    return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)


if __name__ == '__main__':
    vp = VolatilityPricer()
    vp.getRealizedVol()  # 求股票的已实现波动率S
    {'SPY': 0.086197389793546381}
    vp.getRealizedVol(startDate=datetime.date(2018, 1, 1))
    {'SPY': 0.16562165494524139}

    vp = VolatilityPricer()
    vp.getImpliedVol()  # 求期权的隐含波动率