#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:00:48 2018
计算中性策略收益率
@author: lion95
"""

from __future__ import division
from backtest_func import *
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


if __name__ == '__main__':
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    kind = ['btcusdt', 'ethusdt']
    switch = 1
    if switch == 0:
        period = '1440m'
        n = 1440
    else:
        period = '240m'
        n = 240
    now = datetime.datetime.now()
    N_lst = [i for i in range(3, 50, 3)]  # 计算beta的参数
    M_lst = [i for i in range(3, 50, 3)]  # 调仓周期
    p2_lst = [30, 60, 120, 90]  # 平滑周期
    # N_lst = [6]  # 计算beta的参数
    # M_lst = [25]  # 调仓周期
    # p2_lst = [10]  # 平滑周期
    df_btc = pd.read_csv('data/btcusdt_' + period + '.csv').loc[
             :, ['date_time', 'open', 'high', 'low', 'close']]\
        .rename(columns={'open': 'open_b', 'high': 'high_b', 'close': 'close_b', 'low': 'low_b'})
    df_eth = pd.read_csv('data/ethusdt_' + period + '.csv').loc[
             :, ['date_time', 'open', 'high', 'low', 'close']]\
        .rename(columns={'open': 'open_e', 'high': 'high_e', 'close': 'close_e', 'low': 'low_e'})
    df_org = df_btc.merge(df_eth, on=['date_time'])\
        .assign(close_b_1=lambda df: df.close_b.shift(1))\
        .assign(close_e_1=lambda df: df.close_e.shift(1))
    fee = 0.0036
    state_lst = []

    for N in N_lst:
        df_ = df_org.assign(chg_N_btc=lambda df: (df.close_b - df.close_b.shift(N)) / df.close_b.shift(N))\
            .assign(chg_N_eth=lambda df: (df.close_e - df.close_e.shift(N)) / df.close_e.shift(N))\
            .assign(beta=lambda df: df.chg_N_eth / df.chg_N_btc)\
            .assign(beta_1=lambda df: df.beta.shift(1))
        for p2 in p2_lst:
            df = df_.assign(ma_btc=lambda df: tb.MA(df['close_b'].values, p2, matype=0)) \
                .assign(ma_btc_c=lambda df: df.close_b - df.ma_btc)\
                .assign(ma_eth=lambda df: tb.MA(df['close_e'].values, p2, matype=0)) \
                .assign(ma_eth_c=lambda df: df.close_e - df.ma_eth)\
                .assign(ma_btc_chg_1=lambda df: df.ma_btc.shift(1) - df.ma_btc.shift(2)) \
                .assign(ma_eth_chg_1=lambda df: df.ma_eth.shift(1) - df.ma_eth.shift(2)).dropna()
            df.to_csv('data/btceth.csv')
            for M in M_lst:
                print(M)
                group_ = df[df['date_time'] >= '2017-01-01'].reset_index(drop=True)
                back_stime = group_.at[0, 'date_time']
                back_etime = group_.at[len(group_) - 1, 'date_time']
                signal_lst = []
                trad_times = 0
                net_lst = []
                pos_lst = []
                net = 1
                num = 0
                position = 0  # 0:空仓 1：多ETH 空BTC 2：空ETH 多BTC 3:多 BTC 空ETH 4:多ETH 空BTC
                value_b = 10000 / group_.at[0, 'close_b']
                value_e = 10000 / group_.at[0, 'close_b']
                value = value_b + value_e
                value_lst = []
                value_lst_a = []
                value_lst_b = []

                for idx, row_ in group_.iterrows():

                    if position == 0:
                        if (row_['ma_eth_chg_1'] > 0) & (abs(row_['beta_1']) > 1) & (row_['ma_btc_chg_1'] > 0):
                            position = 1
                            s_time = row_.date_time
                            trad_times = trad_times + 1
                            value_start = value
                            k_price_eth = row_.open_e * (1 + fee)
                            k_price_btc = row_.open_b * (1 - fee)
                            cont_btc = -int(value * row_.open_b / 2)
                            cont_eth = int(value * 1000000 / row_.open_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / row_.close_b)
                            value_e = value_e + cont_eth * (row_.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            num = 1
                        elif (row_['ma_eth_chg_1'] < 0) & (abs(row_['beta_1']) > 1) & (row_['ma_btc_chg_1'] < 0):
                            position = 2
                            s_time = row_.date_time
                            trad_times = trad_times + 1
                            value_start = value
                            k_price_eth = row_.open_e * (1 - fee)
                            k_price_btc = row_.open_b * (1 + fee)
                            cont_btc = int(value * row_.open_b / 2)
                            cont_eth = -int(value * 1000000 / row_.open_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / row_.close_b)
                            value_e = value_e + cont_eth * (row_.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            num = 1
                        elif (row_['ma_eth_chg_1'] > 0) & (abs(row_['beta_1']) < 1) & (row_['ma_btc_chg_1'] > 0):
                            position = 3
                            s_time = row_.date_time
                            trad_times = trad_times + 1
                            value_start = value
                            k_price_eth = row_.open_e * (1 - fee)
                            k_price_btc = row_.open_b * (1 + fee)
                            cont_btc = int(value * row_.open_b / 2)
                            cont_eth = -int(value * 1000000 / row_.open_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / row_.close_b)
                            value_e = value_e + cont_eth * (row_.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            num = 1
                        elif (row_['ma_eth_chg_1'] < 0) & (abs(row_['beta_1']) < 1) & (row_['ma_btc_chg_1'] < 0):
                            position = 4
                            s_time = row_.date_time
                            trad_times = trad_times + 1
                            value_start = value
                            k_price_eth = row_.open_e * (1 + fee)
                            k_price_btc = row_.open_b * (1 - fee)
                            cont_btc = -int(value * row_.open_b / 2)
                            cont_eth = int(value * 1000000 / row_.open_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / row_.close_b)
                            value_e = value_e + cont_eth * (row_.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            num = 1
                        else:
                            value = value
                    else:
                        num += 1
                        if (position == 1) and (row_['ma_btc_c'] < 0) and (row_['ma_eth_c'] < 0):

                            e_time = row_.date_time
                            if cont_btc > 0:
                                p_price_btc = row_.close_b * (1 - fee)
                                p_price_eth = row_.close_e * (1 + fee)
                            else:
                                p_price_btc = row_.close_b * (1 + fee)
                                p_price_eth = row_.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / row_.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - row_.close_e_1) / 1000000
                            value = value_b + value_e

                            value_end = value
                            ret = value_end / value_start - 1
                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(ret)
                            signal_row.append(num)
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = 0
                            num = 0
                            # if (row_['ma_eth_chg'] > 0) & (abs(row_['beta']) > 1) & (row_['ma_btc_chg'] > 0):
                            #     position = 1
                            #     trad_times = trad_times + 1
                            #     k_price_eth = row_.open_e * (1 + fee)
                            #     k_price_btc = row_.open_b * (1 - fee)
                            #     cont_btc = -int(value / 2)
                            #     cont_eth = int(abs(cont_btc) * 1000000 / row_.open_b / row_.open_e)
                            #     value_b = value_b + cont_btc * (1 / k_price_btc - 1 / row_.close_b)
                            #     value_e = value_e + cont_eth * (row_.close_e - k_price_eth) / 1000000
                            #     value = value_b + value_e
                            #     num = 1
                        elif (position == 3) and (row_['ma_btc_c'] < 0) and (row_['ma_eth_c'] < 0):
                            e_time = row_.date_time
                            if cont_btc > 0:
                                p_price_btc = row_.close_b * (1 - fee)
                                p_price_eth = row_.close_e * (1 + fee)
                            else:
                                p_price_btc = row_.close_b * (1 + fee)
                                p_price_eth = row_.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / row_.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - row_.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(ret)
                            signal_row.append(num)
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = 0
                            num = 0
                        elif ((position == 2) | (position == 4)) and ((row_['ma_btc_c'] > 0) and (row_['ma_eth_c'] > 0)):
                            e_time = row_.date_time
                            if cont_btc > 0:
                                p_price_btc = row_.close_b * (1 - fee)
                                p_price_eth = row_.close_e * (1 + fee)
                            else:
                                p_price_btc = row_.close_b * (1 + fee)
                                p_price_eth = row_.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / row_.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - row_.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(ret)
                            signal_row.append(num)
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = 0
                            num = 0
                        elif num == M:
                            e_time = row_.date_time
                            if cont_btc > 0:
                                p_price_btc = row_.close_b * (1 - fee)
                                p_price_eth = row_.close_e * (1 + fee)
                            else:
                                p_price_btc = row_.close_b * (1 + fee)
                                p_price_eth = row_.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / row_.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - row_.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(ret)
                            signal_row.append(num)
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = 0
                            num = 0
                        else:
                            value_b = value_b + cont_btc * (1 / row_.close_b_1 - 1 / row_.close_b)
                            value_e = value_e + cont_eth * (row_.close_e - row_.close_e_1) / 1000000
                            value = value_b + value_e
                    pos_lst.append(position)
                    value_lst.append(value)
                if len([i for i in value_lst if i < 0]) > 0:
                    continue
                net_lst = [i/value_lst[0] for i in value_lst]
                # print(net_lst)
                print(N, M, p2)
                # net_df = pd.DataFrame({'value': value_lst,
                #                        'net': net_lst,
                #                        'date_time': group_.date_time.tolist(),
                #                        'close_b': group_.close_b.tolist(),
                #                        'close_e': group_.close_e.tolist(),
                #                        'pos': pos_lst}).assign(
                #     close_n=lambda df: df.close_b / df.close_b.tolist()[0])
                # net_df.to_csv('data/net_.csv')
                # net_df[['date_time', 'net', 'close_n']].plot(
                #     x='date_time', kind='line', grid=True,
                #     title=period)
                # plt.show()
                signal_state = pd.DataFrame(
                    signal_lst, columns=['s_time', 'e_time', 'cont_btc', 'cont_eth', 'k_price_btc',
                                         'k_price_eth', 'p_price_btc', 'p_price_eth',
                                         'ret', 'hold_day', 'position'])
                # signal_state.to_csv('cl_trend/data/state_neu_strategy.csv')
                win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                ann_ROR = annROR(net_lst, n)
                total_ret = net_lst[-1]
                max_retrace = maxRetrace(net_lst, n)
                sharp = yearsharpRatio(net_lst, n)
                state_row = []
                state_row.append(n)
                state_row.append(win_r)
                state_row.append(odds)
                state_row.append(total_ret - 1)
                state_row.append(ann_ROR)
                state_row.append(sharp)
                state_row.append(max_retrace)
                state_row.append(len(signal_state))
                state_row.append(ave_r)
                state_row.append(signal_state.hold_day.mean())

                # state_row.append(N_ATR)
                # state_row.append(ATR_n)
                state_row.append(N)
                state_row.append(M)
                state_row.append(p2)
                # state_row.append(K2)
                # state_row.append(win_stop)
                # state_row.append(status_day)
                state_row.append(back_stime)
                state_row.append(back_etime)
                state_lst.append(state_row)

    res = pd.DataFrame(state_lst,
                       columns=['period', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                'trade_times', 'ave_r', 'ave_hold_days', 'N', 'M', 'P', 'back_stime', 'back_etime'])
    res.to_csv('cl_trend/data/neu_strategy_' + str(n) + '.csv')
