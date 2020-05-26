#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:26:20 2019
期权数据研究
@author: lion95
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
import datetime
from dataapi import *
import talib
import matplotlib.pyplot as plt
from scipy.stats import norm
from jqdatasdk import *

auth('18610039264', 'zg19491001')


def cal_sig(t, s0, X, mktprice, kind):
    # 设定参数
    r = 0.032  # risk-free interest rate
    #    t=float(30)/365 # time to expire (30 days)
    q = 0  # dividend yield
    S0 = s0  # underlying price 正股价
    #    X=2.2 # strike price 行权价
    #    mktprice=0.18 # market price

    # 用二分法求implied volatility，暂只针对call option
    if kind == 0:
        sigma = 0.3  # initial volatility
        C = P = 0
        upper = 1
        lower = 0
        num = 0
        while (abs(C - mktprice) > 1e-3) & (num < 50):
            #            print(abs(C-mktprice))
            d1 = (np.log(S0 / X) + (r - q + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            C = S0 * np.exp(-q * t) * norm.cdf(d1) - X * np.exp(-r * t) * norm.cdf(d2)
            #            P=X*np.exp(-r*t)*norm.cdf(-d2)-S0*np.exp(-q*t)*norm.cdf(-d1)
            if C - mktprice > 0:
                upper = sigma
                sigma = (sigma + lower) / 2
            else:
                lower = sigma
                sigma = (sigma + upper) / 2
            if sigma < 1e-3:
                sigma = sigma + 1e-1
            num = num + 1
        return sigma  # implied volatility
    elif kind == 1:
        sigma = 0.3  # initial volatility
        C = P = 0
        upper = 1
        lower = 0
        num = 0
        while (abs(P - mktprice) > 1e-3) & (num < 50):
            #            print(abs(P-mktprice))
            d1 = (np.log(S0 / X) + (r - q + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            #            C=S0*np.exp(-q*t)*norm.cdf(d1)-X*np.exp(-r*t)*norm.cdf(d2)
            P = X * np.exp(-r * t) * norm.cdf(-d2) - S0 * np.exp(-q * t) * norm.cdf(-d1)
            #            print(P)
            if P - mktprice > 0:
                upper = sigma
                sigma = (sigma + lower) / 2
            else:
                lower = sigma
                sigma = (sigma + upper) / 2
            if sigma < 1e-3:
                sigma = sigma + 1e-1
            num = num + 1
        #            print(sigma)
        #            print('+++++')
        return sigma  # implied volatility


# 获取交易日
def tradeday(sday, eday):
    """
    输入 开始时间 和 截止时间
    输出 list 交易日 datetime格式
    """
    return get_trade_days(sday, eday)


def cal_dis(row):
    CO = cal_sig(row['time_delt'], row['etf_close'], row['exercise_price'], row['close_CO'], 0)
    #    print('______')
    PO = cal_sig(row['time_delt'], row['etf_close'], row['exercise_price'], row['close_PO'], 1)
    return CO * row['position_CO'] - PO * row['position_PO']


if __name__ == '__main__':

    sday = '2018-01-01'
    eday = '2019-04-10'
    tradedays = tradeday(sday, eday)
    tradeday_lst = [i.strftime('%Y-%m-%d') for i in tradedays]

    q = query(opt.OPT_CONTRACT_INFO).filter(opt.OPT_CONTRACT_INFO.exercise_date < '2019-04-01')
    df1 = opt.run_query(q)
    #    print(df)
    #    df.to_excel('option_contet.xls')
    df = df1[['code', 'name', 'contract_type', 'exchange_code', 'underlying_type', 'exercise_price'
        , 'list_date', 'expire_date']]
    df = df.query("exchange_code=='XSHG' & underlying_type=='ETF'")
    df.list_date = df.list_date.apply(lambda s: str(s))
    df.expire_date = df.expire_date.apply(lambda s: str(s))

    # 交易期权列表
    option_lst = df.code.tolist()
    for i in range(len(option_lst)):
        print(option_lst[i])
        q = query(opt.OPT_DAILY_PRICE.code,
                  opt.OPT_DAILY_PRICE.date,
                  opt.OPT_DAILY_PRICE.high,
                  opt.OPT_DAILY_PRICE.open,
                  opt.OPT_DAILY_PRICE.low,
                  opt.OPT_DAILY_PRICE.close,
                  opt.OPT_DAILY_PRICE.change_pct_close,
                  opt.OPT_DAILY_PRICE.volume,
                  opt.OPT_DAILY_PRICE.position).filter(opt.OPT_DAILY_PRICE.code == option_lst[i]).order_by(
            opt.OPT_DAILY_PRICE.date.desc()).limit(3000)
        temp = opt.run_query(q)
        if i == 0:
            op_price = temp.copy()
        else:
            op_price = pd.concat([op_price, temp.copy()])
    op_price.to_csv('data/op_price_day_{var1}_{var2}.csv'.format(var1=sday, var2=eday))
    op_price = op_price.rename(columns={'date': 'day'})
    op_price.day = op_price.day.apply(lambda s: str(s)[:10])
    # 获取50ETF的数据
    price_50ETF = stock_price('510050.XSHG', '1d', sday, eday)
    price_50ETF.tradedate = price_50ETF.tradedate.apply(lambda s: str(s)[:10])
    price_50ETF_s = price_50ETF[['tradedate', 'close']]
    price_50ETF_s.columns = ['day', 'etf_close']

    ret = list()
    for d in tradeday_lst:
        print(d)
        res = list()
        temp = df.query("list_date<='{var1}' & expire_date>='{var1}'".format(var1=d)).copy()
        temp_CO = temp[temp['contract_type'] == 'CO']
        temp_CO = temp_CO[['code', 'exercise_price', 'list_date', 'expire_date']]
        temp_CO.columns = ['code_CO', 'exercise_price', 'list_date', 'expire_date']
        temp_PO = temp[temp['contract_type'] == 'PO']
        temp_PO = temp_PO[['code', 'exercise_price', 'list_date', 'expire_date']]
        temp_PO.columns = ['code_PO', 'exercise_price', 'list_date', 'expire_date']

        data_heyue = temp_CO.merge(temp_PO, on=['exercise_price', 'list_date', 'expire_date'])
        data_heyue['day'] = d
        data_heyue['time_delt'] = pd.to_datetime(data_heyue['expire_date']) - pd.to_datetime(data_heyue['day'])
        data_heyue.time_delt = data_heyue.time_delt.apply(lambda s: s.days)
        data_heyue = data_heyue.merge(price_50ETF_s, on=['day'])
        op_price_1 = op_price.copy()
        op_price_1 = op_price_1.rename(columns={'code': 'code_CO'})
        data_heyue1 = data_heyue.merge(op_price_1, on=['code_CO', 'day'])
        data_heyue1 = data_heyue1.rename(columns={'close': 'close_CO', 'position': 'position_CO'})
        op_price_2 = op_price.copy()
        op_price_2 = op_price_2.rename(columns={'code': 'code_PO'})
        data_heyue2 = data_heyue1.merge(op_price_2, on=['code_PO', 'day'])
        data_heyue2 = data_heyue2.rename(columns={'close': 'close_PO', 'position': 'position_PO'})

        data_heyue2['time_delt'] = data_heyue2.time_delt / 365
        vol_sum = data_heyue2.position_CO.sum() + data_heyue2.position_PO.sum()
        data_heyue2['position_CO'] = data_heyue2.position_CO / vol_sum
        data_heyue2['position_PO'] = data_heyue2.position_PO / vol_sum

        data_heyue2['op_var_dis'] = [cal_dis(row) for idx, row in data_heyue2.iterrows()]

        res.append(d)
        res.append(data_heyue2.op_var_dis.sum())
        ret.append(res)
    TTM = pd.DataFrame(ret)
    TTM.columns = ['day', 'ttm']
    TTM = TTM.merge(price_50ETF_s)

    #    fig = plt.figure(figsize=(25,15))
    #    ax1 = plt.subplot(2,1,1)
    #    ax1.plot(TTM.etf_close.tolist(),label='etf')
    #    ax1.grid()
    #    ax1.set_title('{kind}_price'.format(kind='etf'))
    #
    #    ax2 = plt.subplot(2,1,2)
    #    ax2.plot(TTM.ttm.tolist(),label='ttm')

    # =============================================================================
    # 另一种作图
    # =============================================================================
    #    TTM.day=TTM.day.apply(lambda s:s[:16])
    TTM.day = TTM.day.apply(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))

    TTM = TTM[['day', 'etf_close', 'ttm']]

    TTM['day'] = pd.to_datetime(TTM['day'])
    TTM = TTM.set_index(['day'], drop=True)

    ax = TTM.plot(secondary_y=['etf_close'], title='etf', use_index=True, figsize=(25, 15), legend=2)
    ax.grid(True, linestyle="-.")
    ax.grid(which='minor', axis='both')
