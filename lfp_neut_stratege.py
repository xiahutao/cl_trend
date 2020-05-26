#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:00:48 2018
计算中性策略收益率
@author: lion95
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
import datetime
from dataapi import *
import talib as tb
import matplotlib.pyplot as plt
import math


def GarmanKlassYang_Vol(data, n=1440):
    a = 0.5 * np.log(data['high'][1:] / data['low'][1:]) ** 2
    b = (2 * np.log(2) - 1) * (np.log(data['close'][1:] / data['open'][1:]) ** 2)
    c_array = np.log(data['open'][1:].values / data['close'][:-1].values) ** 2
    c = pd.Series(c_array, index=list(a.index))
    vol = np.sqrt(sum(a - b + c) / (len(data) - 1)) * np.sqrt(250 * 1440 / n)
    return vol


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

    kind = ['btcusdt', 'ethusdt']
    switch = 0
    #    N_lst=range(8,40,2) #计算beta的参数
    #    M_lst=range(5,15)# 调仓周期
    if switch == 0:
        period = '1d'
    else:
        period = '4h'
    now = datetime.datetime.now()
    start = (now - datetime.timedelta(days=1000)).strftime('%Y-%m-%d')
    end = now.strftime('%Y-%m-%d %H:%M')[:10]
    N_lst = [14]  # 计算beta的参数
    M_lst = [11]  # 调仓周期
    p2_lst = [15]  # 平滑周期
    df_btc = get_exsymbol_kline('BITFINEX', kind[0], period, start, end)[2]
    df_btc.to_csv('data/btc_price.csv')
    df_eth = get_exsymbol_kline('BITFINEX', kind[1], period, start, end)[2]
    df_btc.date = df_btc.date.apply(lambda s: s[:10])
    df_eth.date = df_eth.date.apply(lambda s: s[:10])
    df_btc = df_btc.fillna(method='ffill')
    df_eth = df_eth.fillna(method='ffill')
    result_lst = list()
    for N in N_lst:
        for M in M_lst:
            for p2 in p2_lst:
                r_lst = list()
                method = 'neut_strategy' + '_' + str(N) + '_' + str(M) + '_' + str(p2)
                print(method)
                df_btc = df_btc.loc[:, ['date', 'open', 'high', 'low', 'close']] \
                    .assign(ma_btc=lambda df: tb.MA(df['close'].values, p2, matype=0)) \
                    .assign(ma_btc_c=lambda df: df.close - df.ma_btc) \
                    .assign(chg_btc=lambda df: (df.close - df.close.shift(1)) / df.close.shift(1)) \
                    .assign(chg_N_btc=lambda df: (df.close - df.close.shift(N)) / df.close.shift(N))

                df_eth = df_eth.loc[:, ['date', 'close']] \
                    .assign(ma_eth=lambda df: tb.MA(df['close'].values, p2, matype=0)) \
                    .assign(ma_eth_c=lambda df: df.close - df.ma_eth) \
                    .assign(chg_eth=lambda df: (df.close - df.close.shift(1)) / df.close.shift(1)) \
                    .assign(chg_N_eth=lambda df: (df.close - df.close.shift(N)) / df.close.shift(N))

                df = df_btc.merge(df_eth, on='date')
                df = df.dropna()

                df['beta'] = df['chg_N_eth'] / df['chg_N_btc']

                group_ = df[['date', 'open', 'high', 'low', 'close_x', 'ma_btc', 'ma_eth', 'chg_btc', 'chg_eth', 'beta',
                             'ma_btc_c', 'ma_eth_c']]
                group_ = group_.rename(columns={'close_x': 'close'})
                group_ = group_.assign(ma_btc_chg=lambda df: df.ma_btc - df.ma_btc.shift(1)) \
                    .assign(ma_eth_chg=lambda df: df.ma_eth - df.ma_eth.shift(1)) \
                    .assign(ma_eth_chg_1=lambda df: df.ma_eth_chg.shift(1)) \
                    .assign(ma_btc_chg_1=lambda df: df.ma_btc_chg.shift(1)) \
                    .assign(beta_1=lambda df: df.beta.shift(1)) \
                    .assign(ma_btc_c_1=lambda df: df.ma_btc_c.shift(1)) \
                    .assign(ma_eth_c_1=lambda df: df.ma_eth_c.shift(1))

                group_.index = range(len(group_))

                #                GarmanKlassYang_Vol_list=list()
                #                for idx,_row in group_.iterrows():
                #                    if idx<p1:
                #                        GarmanKlassYang_Vol_list.append(0)
                #                    else:
                #                        temp=group_.iloc[(idx-p1):idx,:]
                #                        GarmanKlassYang_Vol_list.append(GarmanKlassYang_Vol(temp))
                #
                #                group_=group_.assign(GarmanKlassYang_Vol=GarmanKlassYang_Vol_list)\
                #                             .assign(GarmanKlassYang_Vol_1=lambda df:df.GarmanKlassYang_Vol.shift(1))
                #                group_.index=range(len(group_))
                curve_lst = list()
                pos_lst = list()
                num = 0
                trade_nums = 0
                position = 0  # 0:空仓 1：多ETH 空BTC 2：空ETH 多BTC 3:多 BTC 空ETH 4:多ETH 空BTC

                for idx, row_ in group_.iterrows():
                    #        print(num%5)
                    #       if (position==0) and (row_['GarmanKlassYang_Vol_1']>0.0) :
                    if (position == 0):
                        if (row_['ma_eth_chg_1'] > 0) & (abs(row_['beta_1']) > 1) & (row_['ma_btc_chg_1'] > 0):
                            position = 1
                            trade_nums = trade_nums + 1
                        elif (row_['ma_eth_chg_1'] < 0) & (abs(row_['beta_1']) > 1) & (row_['ma_btc_chg_1'] < 0):
                            position = 2
                            trade_nums = trade_nums + 1
                        elif (row_['ma_eth_chg_1'] > 0) & (abs(row_['beta_1']) < 1) & (row_['ma_btc_chg_1'] > 0):
                            position = 3
                            trade_nums = trade_nums + 1
                        elif (row_['ma_eth_chg_1'] < 0) & (abs(row_['beta_1']) < 1) & (row_['ma_btc_chg_1'] < 0):
                            position = 4
                            trade_nums = trade_nums + 1
                    else:
                        if (num % M) == 0:
                            #                print('************')
                            position = 0
                            num = 0
                        else:
                            if position in [1, 3]:
                                if (row_['ma_btc_c_1'] < 0) and (row_['ma_eth_c_1'] < 0):
                                    position = 0
                                    num = 0
                            elif position in [2, 4]:
                                if (row_['ma_btc_c_1'] > 0) and (row_['ma_eth_c_1'] > 0):
                                    position = 0
                                    num = 0

                    if position == 0:
                        curve_lst.append(0)
                    elif position in [1, 4]:
                        curve_lst.append((row_['chg_eth'] - row_['chg_btc']) / 2)
                        num = num + 1
                    elif position in [2, 3]:
                        curve_lst.append((row_['chg_btc'] - row_['chg_eth']) / 2)
                        num = num + 1
                    pos_lst.append(position)
                group_ = group_.assign(chg_neu=curve_lst) \
                    .assign(position=pos_lst) \
                    .assign(curve_neu=lambda df: df.chg_neu + 1) \
                    .assign(curve_neu=lambda df: df.curve_neu.cumprod())
                group_['date'] = pd.to_datetime(group_['date'])
                group_[['date', 'curve_neu']].plot(x='date', y='curve_neu')
                group_.to_csv('neu_stra.csv')

                # 统计指标
                res_lst = group_.curve_neu.tolist()
                r_lst.append((res_lst[-1] - res_lst[0]) / res_lst[0])
                r_lst.append(annROR(res_lst))
                r_lst.append(maxRetrace(res_lst))
                r_lst.append(yearsharpRatio(res_lst))
                r_lst.append(trade_nums)
                r_lst.append((res_lst[-1] - res_lst[0]) / (res_lst[0] * trade_nums))
                r_lst.append(method)
                result_lst.append(r_lst)
    res = pd.DataFrame(result_lst,
                       columns=['ror', 'annror', 'maxRetrace', 'yearsharpRatio', 'trade_nums', 'avg_ror', 'method'])
    #                print([(res_lst[-1]-res_lst[0])/res_lst[0],annROR(res_lst),maxRetrace(res_lst),yearsharpRatio(res_lst)])
    res.to_csv('result/neu_strategy_1.csv')
