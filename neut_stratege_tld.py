#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:16:35 2019
中性策略 屠龙刀
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
    end = '2019-02-20'
    os.chdir(path)
    #    M_s=np.arange(0,1,0.1) #平仓阈值
    #    N_s=range(5,11,2) #标准化窗口
    #    P_s=np.arange(1,2,0.2) # 阈值

    #    N1_lst=[5]   # 短均线参数
    #    N2_lst=range(10,60,5)  # 中均线参数
    #    N3_lst=range(100,300,50) # 长均线参数,敏感
    #    N4_lst=[10] # 唐奇安通道数

    N1_lst = [4]  # 短均线参数
    N2_lst = [19]  # 中均线参数
    N3_lst = [200]  # 长均线参数,敏感
    N4_lst = [6]  # 唐奇安通道数

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

    for N1 in N1_lst:
        for N2 in N2_lst:
            for N3 in N3_lst:
                for N4 in N4_lst:
                    r_lst = list()
                    e_lst = list()
                    date_lst = list()
                    method = 'neut_stratege_' + str(N1) + '_' + str(N2) + '_' + str(N3) + '_' + str(N4)
                    print(method)
                    ethbtc_price = eth_price_btc[['date', 'close']]
                    ethbtc_price = ethbtc_price.dropna()
                    ethbtc_price.columns = ['date', 'eth_btc']
                    ethbtc_price.date = ethbtc_price.date.apply(lambda s: s[:10])
                    ethbtc_price = ethbtc_price.assign(ma1_ethbtc=lambda df: tb.MA(df['eth_btc'].values, N1)) \
                        .assign(ma2_ethbtc=lambda df: tb.MA(df['eth_btc'].values, N2)) \
                        .assign(ma3_ethbtc=lambda df: tb.MA(df['eth_btc'].values, N3)) \
                        .assign(hhv_ethbtc=lambda df: tb.MAX(df['eth_btc'].values, N4)) \
                        .assign(llv_ethbtc=lambda df: tb.MIN(df['eth_btc'].values, N4)) \
                        .assign(ethbtc_1=lambda df: df.eth_btc.shift(1)) \
                        .assign(ma1_ethbtc_1=lambda df: df.ma1_ethbtc.shift(1)) \
                        .assign(ma2_ethbtc_1=lambda df: df.ma2_ethbtc.shift(1)) \
                        .assign(ma3_ethbtc_1=lambda df: df.ma3_ethbtc.shift(1)) \
                        .assign(hhv_ethbtc_2=lambda df: df.hhv_ethbtc.shift(2)) \
                        .assign(llv_ethbtc_2=lambda df: df.llv_ethbtc.shift(2))

                    data = btc_price.merge(eth_price)
                    data = data.merge(ethbtc_price)
                    data = data.dropna()

                    curve_lst = list()
                    pos_lst = list()
                    num = 0
                    trade_nums = 0
                    position = 0  # 0:空仓 1：多ETH 空BTC 2：空ETH 多BTC
                    p_lst = list()
                    for idx, row_ in data.iterrows():
                        #        print(num%5)

                        #       if (position==0) and (row_['GarmanKlassYang_Vol_1']>0.0) :
                        if (position == 0):
                            if (row_['ma1_ethbtc_1'] > row_['ma2_ethbtc_1']) & (
                                    row_['ethbtc_1'] > row_['hhv_ethbtc_2']) & (
                                    row_['ethbtc_1'] > row_['ma3_ethbtc_1']):
                                position = 1
                                trade_nums = trade_nums + 1
                            elif (row_['ma1_ethbtc_1'] < row_['ma2_ethbtc_1']) & (
                                    row_['ethbtc_1'] < row_['llv_ethbtc_2']) & (
                                    row_['ethbtc_1'] < row_['ma3_ethbtc_1']):
                                position = 2
                                trade_nums = trade_nums + 1
                        else:
                            if (position == 1) & (row_['ethbtc_1'] < row_['llv_ethbtc_2']):
                                #                print('************')
                                position = 0
                            elif (position == 2) & (row_['ethbtc_1'] > row_['hhv_ethbtc_2']):
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
                    data = data.assign(chg_neu=curve_lst) \
                        .assign(position=pos_lst) \
                        .assign(curve_neu=lambda df: df.chg_neu + 1) \
                        .assign(curve_neu=lambda df: df.curve_neu.cumprod()) \
                        .assign(position=p_lst)
                    #                    data['date'] = pd.to_datetime(data['date'])
                    #                    fig, ax = plt.subplots(figsize=(20,10))
                    #                    Loc = mdates.DayLocator(interval=50)
                    #                    ax.xaxis.set_major_locator(Loc)
                    #                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
                    #
                    #                    data[['date','curve_neu']].plot(x='date',ax=ax)
                    #                    data.to_csv('neu_stra_tld.csv')

                    # 统计指标
                    res_lst = data.curve_neu.tolist()
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
#    res.to_csv('result/neu_strategy_tld.csv')
