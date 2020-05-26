#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:16:35 2019
统计套利分析 基于协整分析
@author: lion95
"""

from __future__ import division
import pandas as pd
import numpy as np
from dataapi import *
import os
import talib as tb
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import matplotlib.dates as mdates


# 输入是一DataFrame，每一列是一支股票在每一日的价格
# def find_cointegrated_pairs(dataframe):
#    # 得到DataFrame长度
#    n = dataframe.shape[1]
#    # 初始化p值矩阵
#    pvalue_matrix = np.ones((n, n))
#    # 抽取列的名称
#    keys = dataframe.keys()
#    # 初始化强协整组
#    pairs = []
#    # 对于每一个i
#    for i in range(n):
#        # 对于大于i的j
#        for j in range(i+1, n):
#            # 获取相应的两只股票的价格Series
#            stock1 = dataframe[keys[i]]
#            stock2 = dataframe[keys[j]]
#            # 分析它们的协整关系
#            result = sm.tsa.stattools.coint(stock1, stock2)
#            # 取出并记录p值
#            pvalue = result[1]
#            pvalue_matrix[i, j] = pvalue
#            # 如果p值小于0.05
#            if pvalue < 0.05:
#                # 记录股票对和相应的p值
#                pairs.append((keys[i], keys[j], pvalue))
#    # 返回结果
#    return pvalue_matrix, pairs

def maxRetrace(list):
    '''
    :param list:netlist
    :return: 最大历史回撤
    '''
    Max = 0
    for i in range(len(list)):
        if 1 - list[i] / max(list[:i + 1]) > Max:
            Max = 1 - list[i] / max(list[:i + 1])

    return Max


def annROR(netlist, n=1440):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 365 * 1440 / len(netlist) / n) - 1


def daysharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row)


def yearsharpRatio(netlist, n=1440):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(365 * 1440 / n, 0.5)


if __name__ == '__main__':
    path = r'/Users/lion95/Documents/mywork/张芳的代码'
    start = '2016-01-01'
    end = '2019-01-22'
    os.chdir(path)
    #    M_s=np.arange(0,1,0.1) #平仓阈值
    #    N_s=range(5,11,2) #标准化窗口
    #    P_s=np.arange(1,2,0.2) # 阈值

    M_s = [19]  # 平仓阈值
    N_s = [5]  # 标准化窗口
    P_s = [1.4]  # 开仓阈值

    #    M_s=range(5,30) #ethbtc均线
    #    N_s=range(5,15,2) #标准化窗口
    #    P_s=np.arange(1.4,2,0.2) #开仓阈值

    #    data=pd.read_csv('data/position_l.csv',index_col=0)
    btc_price = get_exsymbol_kline('BITFINEX', 'btcusdt', '1d', start, end)[2]
    eth_price_usdt = get_exsymbol_kline('BITFINEX', 'ethusdt', '1d', start, end)[2]
    eth_price_btc = get_exsymbol_kline('BITFINEX', 'ethbtc', '1d', start, end)[2]

    btc_price = btc_price[['date', 'close']]
    btc_price.columns = ['date', 'btc_usdt']
    btc_price.date = btc_price.date.apply(lambda s: s[:10])
    btc_price['btc_chg'] = (btc_price['btc_usdt'] - btc_price['btc_usdt'].shift(1)) / btc_price['btc_usdt'].shift(1)

    eth_price = eth_price_usdt[['date', 'close']]
    eth_price.columns = ['date', 'eth_usdt']
    eth_price.date = eth_price.date.apply(lambda s: s[:10])
    eth_price['eth_chg'] = (eth_price['eth_usdt'] - eth_price['eth_usdt'].shift(1)) / eth_price['eth_usdt'].shift(1)

    tradelist = eth_price.date.tolist()

    result_lst = list()

    for M in M_s:
        for N in N_s:
            for P in P_s:
                r_lst = list()
                e_lst = list()
                date_lst = list()
                method = 'neut_stratege_' + str(M) + '_' + str(N) + '_' + str(P)
                print(method)
                ethbtc_price = eth_price_btc[['date', 'close']]
                ethbtc_price = ethbtc_price.dropna()
                ethbtc_price.columns = ['date', 'eth_btc']
                ethbtc_price.date = ethbtc_price.date.apply(lambda s: s[:10])
                ethbtc_price = ethbtc_price.assign(ma_ethbtc=lambda df: tb.MA(df['eth_btc'].values, M)) \
                    .assign(ma_ethbtc_1=lambda df: df.ma_ethbtc.shift(1)) \
                    .assign(ethbtc_1=lambda df: df.eth_btc.shift(1))
                data = btc_price.merge(eth_price)
                data = data.merge(ethbtc_price)
                data = data.dropna()
                for i in range(len(tradelist))[240:]:
                    stime = tradelist[i - 240]
                    etime = tradelist[i]
                    temp = data.query("date>='{var1}' & date<='{var2}'".format(var1=stime, var2=etime))
                    x = temp['btc_usdt']
                    y = temp['eth_usdt']
                    X = sm.add_constant(x)
                    result = (sm.OLS(y, X)).fit()
                    k = result.params[1]
                    b = result.params[0]
                    e_value = y - result.fittedvalues
                    e_lst.append(e_value.iloc[-1])
                    date_lst.append(etime)
                #    plt.plot(e_lst)
                sinal_lst = list()
                sinal_lst.append(date_lst)
                sinal_lst.append(e_lst)
                sinal = pd.DataFrame(sinal_lst).T
                sinal.columns = ['date', 'e_value']
                sinal.e_value = sinal.e_value.astype('float')
                sinal = sinal.assign(m=lambda df: tb.MA(df['e_value'].values, N)) \
                    .assign(d=lambda df: tb.STDDEV(df['e_value'].values, timeperiod=N, nbdev=1)) \
                    .assign(e_v_n=lambda df: (df.e_value - df.m) / df.d)
                sinal = sinal.dropna()
                #    sinal.e_v_n.plot()
                df = data.merge(sinal[['date', 'e_v_n']])
                df['e_v_n_1'] = df.e_v_n.shift(1)
                #                df[['date','e_v_n_1']].plot(x='date')

                df = df.dropna()

                curve_lst = list()
                pos_lst = list()
                num = 0
                trade_nums = 0
                position = 0  # 0:空仓 1：多ETH 空BTC 2：空ETH 多BTC
                p_lst = list()
                for idx, row_ in df.iterrows():
                    #        print(num%5)

                    #       if (position==0) and (row_['GarmanKlassYang_Vol_1']>0.0) :
                    if (position == 0):
                        if (row_['e_v_n_1'] > P) & (row_['ethbtc_1'] > row_['ma_ethbtc_1']):
                            position = 1
                            trade_nums = trade_nums + 1
                        elif (row_['e_v_n_1'] < -P) & (row_['ethbtc_1'] < row_['ma_ethbtc_1']):
                            position = 2
                            trade_nums = trade_nums + 1
                    else:
                        if (position == 1) & (row_['e_v_n_1'] < 0):
                            #                print('************')
                            position = 0
                        elif (position == 2) & (row_['e_v_n_1'] > 0):
                            position = 0

                    p_lst.append(position)
                    if position == 0:
                        curve_lst.append(0)
                    elif position in [1]:
                        curve_lst.append((row_['eth_chg'] - row_['btc_chg']) / 2)
                    #            num=num+1
                    elif position in [2]:
                        curve_lst.append((row_['btc_chg'] - row_['eth_chg']) / 2)
                    #            num=num+1
                    pos_lst.append(position)
                df = df.assign(chg_neu=curve_lst) \
                    .assign(position=pos_lst) \
                    .assign(curve_neu=lambda df: df.chg_neu + 1) \
                    .assign(curve_neu=lambda df: df.curve_neu.cumprod()) \
                    .assign(position=p_lst)
                df['date'] = pd.to_datetime(df['date'])
                fig, ax = plt.subplots(figsize=(20, 10))
                Loc = mdates.DayLocator(interval=50)
                ax.xaxis.set_major_locator(Loc)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))

                df[['date', 'curve_neu']].plot(x='date', ax=ax)
                df.to_csv('neu_stra_xiezheng.csv')

                # 统计指标
                res_lst = df.curve_neu.tolist()
                r_lst.append((res_lst[-1] - res_lst[0]) / res_lst[0])
                r_lst.append(annROR(res_lst))
                r_lst.append(maxRetrace(res_lst))
                r_lst.append(yearsharpRatio(res_lst))
                r_lst.append(trade_nums)
                if trade_nums > 0:
                    r_lst.append((res_lst[-1] - res_lst[0]) / (res_lst[0] * trade_nums))
                else:
                    r_lst.append(0)
                r_lst.append(method)
                result_lst.append(r_lst)
    res = pd.DataFrame(result_lst,
                       columns=['ror', 'annror', 'maxRetrace', 'yearsharpRatio', 'trade_nums', 'avg_ror', 'method'])
#                print([(res_lst[-1]-res_lst[0])/res_lst[0],annROR(res_lst),maxRetrace(res_lst),yearsharpRatio(res_lst)])
#    res.to_csv('result/neu_strategy_xiezheng.csv')
