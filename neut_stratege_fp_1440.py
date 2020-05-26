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
    period = '240m'
    n = 240
    now = datetime.datetime.now()
    ma_period_lst = [10]  # 唐奇安通道周期

    beta_period_lst = [i for i in range(3, 22, 3)]
    # beta_period_lst = [10]

    status_days_lst = [0]
    loss_stop_lst = [0.1, 0.15, 0.2, 0.25, 0.05]
    # loss_stop_lst = [0.15]

    df_btc = pd.read_csv('data/btcusdt_1m.csv').loc[
             :, ['tickid', 'open', 'close']] \
        .rename(columns={'open': 'open_b', 'close': 'close_b'}).assign(close_b_1=lambda df: df.close_b.shift(1))
    df_eth = pd.read_csv('data/ethusdt_1m.csv').loc[
             :, ['tickid', 'open', 'close']] \
        .rename(columns={'open': 'open_e', 'close': 'close_e'}).assign(close_e_1=lambda df: df.close_e.shift(1))
    df_org_1m = df_btc.merge(df_eth, on=['tickid']).sort_values(['tickid'])
    del df_btc
    del df_eth
    df_btc = pd.read_csv('data/btcusdt_' + period + '.csv').loc[
             :, ['tickid', 'open', 'close']] \
        .rename(columns={'open': 'open_b', 'close': 'close_b'}).assign(close_b_1=lambda df: df.close_b.shift(1))
    df_eth = pd.read_csv('data/ethusdt_' + period + '.csv').loc[
             :, ['tickid', 'open', 'close']] \
        .rename(columns={'open': 'open_e', 'close': 'close_e'}).assign(close_e_1=lambda df: df.close_e.shift(1))
    df_org = df_btc.merge(df_eth, on=['tickid']).sort_values(['tickid'])
    del df_btc
    del df_eth
    fee = 0.002
    date_lst = [('2017-01-01', '2019-02-30'), ('2017-01-01', '2018-01-01'), ('2018-01-01', '2018-07-01'), ('2018-07-01', '2019-02-30')]
    # date_lst = [('2017-01-01', '2019-02-30')]
    state_lst = []
    for beta_period in beta_period_lst:
        df_ = df_org.assign(chg_N_btc=lambda df: (df.close_b - df.close_b.shift(beta_period)) / df.close_b.shift(beta_period)) \
            .assign(chg_N_eth=lambda df: (df.close_e - df.close_e.shift(beta_period)) / df.close_e.shift(beta_period)) \
            .assign(beta_1=lambda df: abs(df.chg_N_eth.shift(1) / df.chg_N_btc).shift(1))
        for ma_period in ma_period_lst:
            group_ = df_.assign(ma_btc=lambda df: tb.MA(df['close_b'].shift(1).values, ma_period - 1, matype=0)) \
                .assign(ma_eth=lambda df: tb.MA(df['close_e'].shift(1).values, ma_period - 1, matype=0)) \
                .assign(close_b_n=lambda df: df['close_b'].shift(ma_period))\
                .assign(close_e_n=lambda df: df['close_e'].shift(ma_period))\
                .sort_values(['tickid']).fillna(method='ffill') \
                .reset_index(drop=True)[['tickid', 'ma_btc', 'ma_eth', 'close_b_n', 'close_e_n', 'beta_1']] \
                .merge(df_org_1m, on=['tickid'], how='right').sort_values(['tickid']).fillna(method='ffill') \
                .assign(c_ma_b=lambda df: df.close_b - df.ma_btc)\
                .assign(c_ma_e=lambda df: df.close_e - df.ma_eth)\
                .assign(ma_zd_b=lambda df: df.close_b - df.close_b_n)\
                .assign(ma_zd_e=lambda df: df.close_e - df.close_e_n) \
                .assign(date_time=lambda df: df.tickid.apply(lambda x: str(datetime.datetime.fromtimestamp(x)))) \
                .dropna().reset_index(drop=True)
            print(group_)
            # group_.to_csv('cl_trend/data/data.csv')
            for loss_stop in loss_stop_lst:
                print(loss_stop)
                signal_lst = []
                trad_times = 0
                net_lst = []
                pos_lst = []
                net = 1
                num = 0
                status = -100000
                position = 0  # 0:空仓 1：多ETH 空BTC -1：空ETH 多BTC
                value_b = 10000 / group_.at[0, 'open_b']
                value_e = 10000 / group_.at[0, 'open_b']
                value = value_b + value_e
                value_lst = [value]
                value_lst_b = [value_b]
                value_lst_e = [value_e]

                t0 = time.time()
                for idx, _row in group_.iterrows():
                    if position == 0:

                        if (_row.beta_1 > 1) & (_row.c_ma_b > 0) & (_row.c_ma_e > 0) & (_row.ma_zd_b > 0) & \
                                (_row.ma_zd_e > 0):

                            position = 1
                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 + fee)
                            k_price_btc = _row.close_b * (1 - fee)
                            cont_btc = -int(value * _row.close_b / 2)
                            cont_eth = int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (
                                        _row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)

                        elif (_row.beta_1 < 1) & (_row.c_ma_b < 0) & (_row.c_ma_e < 0) & (_row.ma_zd_b < 0) & \
                                (_row.ma_zd_e < 0):
                            position = 1
                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 + fee)
                            k_price_btc = _row.close_b * (1 - fee)
                            cont_btc = -int(value * _row.close_b / 2)
                            cont_eth = int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (
                                        _row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)

                        elif (_row.beta_1 > 1) & (_row.c_ma_b < 0) & (_row.c_ma_e < 0) & (_row.ma_zd_b < 0) & \
                                (_row.ma_zd_e < 0):
                            position = -1
                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 - fee)
                            k_price_btc = _row.close_b * (1 + fee)
                            cont_btc = int(value * _row.close_b / 2)
                            cont_eth = -int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (
                                        _row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)

                        elif (_row.beta_1 < 1) & (_row.c_ma_b > 0) & (_row.c_ma_e > 0) & (_row.ma_zd_b > 0) & \
                                (_row.ma_zd_e > 0):
                            position = -1
                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 - fee)
                            k_price_btc = _row.close_b * (1 + fee)
                            cont_btc = int(value * _row.close_b / 2)
                            cont_eth = -int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (
                                        _row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)

                        else:
                            value = value
                            value_b = value_b
                            value_e = value_e
                    elif position == 1:
                        if (_row.beta_1 < 1) & (_row.c_ma_b > 0) & (_row.c_ma_e > 0) & (_row.ma_zd_b > 0) & \
                                (_row.ma_zd_e > 0):
                            e_time = _row.date_time
                            if cont_btc > 0:
                                p_price_btc = _row.close_b * (1 - fee)
                                p_price_eth = _row.close_e * (1 + fee)
                            else:
                                p_price_btc = _row.close_b * (1 + fee)
                                p_price_eth = _row.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - _row.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            high_value.append(value_end)

                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(value_start)
                            signal_row.append(value_end)
                            signal_row.append(ret)
                            signal_row.append((max(high_value) / value_start) - 1)
                            signal_row.append(len(hold_value))
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = -1
                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 - fee)
                            k_price_btc = _row.close_b * (1 + fee)
                            cont_btc = int(value * _row.close_b / 2)
                            cont_eth = -int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (_row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)
                        elif (_row.beta_1 > 1) & (_row.c_ma_b < 0) & (_row.c_ma_e < 0) & (_row.ma_zd_b < 0) & \
                                (_row.ma_zd_e < 0):
                            e_time = _row.date_time
                            if cont_btc > 0:
                                p_price_btc = _row.close_b * (1 - fee)
                                p_price_eth = _row.close_e * (1 + fee)
                            else:
                                p_price_btc = _row.close_b * (1 + fee)
                                p_price_eth = _row.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - _row.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            high_value.append(value_end)

                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(value_start)
                            signal_row.append(value_end)
                            signal_row.append(ret)
                            signal_row.append((max(high_value) / value_start) - 1)
                            signal_row.append(len(hold_value))
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = -1

                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 - fee)
                            k_price_btc = _row.close_b * (1 + fee)
                            cont_btc = int(value * _row.close_b / 2)
                            cont_eth = -int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (_row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)
                        else:
                            value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (_row.close_e - _row.close_e_1) / 1000000
                            value = value_b + value_e

                            if (_row.tickid + 60 + 28800) % (n * 60) == 0:
                                hold_value.append(value)
                            high_value.append(value)
                            if value <= max(hold_value) * (1-loss_stop):
                                e_time = _row.date_time
                                if cont_btc > 0:
                                    p_price_btc = _row.close_b * (1 - fee)
                                    p_price_eth = _row.close_e * (1 + fee)
                                else:
                                    p_price_btc = _row.close_b * (1 + fee)
                                    p_price_eth = _row.close_e * (1 - fee)
                                value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / p_price_btc)
                                value_e = value_e + cont_eth * (p_price_eth - _row.close_e_1) / 1000000
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
                                signal_row.append(value_start)
                                signal_row.append(value_end)
                                signal_row.append(ret)
                                signal_row.append((max(high_value) / value_start) - 1)
                                signal_row.append(len(hold_value))
                                signal_row.append(position)
                                signal_lst.append(signal_row)
                                position = 0
                    else:
                        if (_row.beta_1 > 1) & (_row.c_ma_b > 0) & (_row.c_ma_e > 0) & (_row.ma_zd_b > 0) & \
                                (_row.ma_zd_e > 0):
                            e_time = _row.date_time
                            if cont_btc > 0:
                                p_price_btc = _row.close_b * (1 - fee)
                                p_price_eth = _row.close_e * (1 + fee)
                            else:
                                p_price_btc = _row.close_b * (1 + fee)
                                p_price_eth = _row.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - _row.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            high_value.append(value_end)

                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(value_start)
                            signal_row.append(value_end)
                            signal_row.append(ret)
                            signal_row.append((max(high_value) / value_start) - 1)
                            signal_row.append(len(hold_value))
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = 1
                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 - fee)
                            k_price_btc = _row.close_b * (1 + fee)
                            cont_btc = int(value * _row.close_b / 2)
                            cont_eth = -int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (_row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)
                        elif (_row.beta_1 < 1) & (_row.c_ma_b < 0) & (_row.c_ma_e < 0) & (_row.ma_zd_b < 0) & \
                                (_row.ma_zd_e < 0):
                            e_time = _row.date_time
                            if cont_btc > 0:
                                p_price_btc = _row.close_b * (1 - fee)
                                p_price_eth = _row.close_e * (1 + fee)
                            else:
                                p_price_btc = _row.close_b * (1 + fee)
                                p_price_eth = _row.close_e * (1 - fee)
                            value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / p_price_btc)
                            value_e = value_e + cont_eth * (p_price_eth - _row.close_e_1) / 1000000
                            value = value_b + value_e
                            value_end = value
                            ret = value_end / value_start - 1
                            high_value.append(value_end)

                            signal_row = []
                            signal_row.append(s_time)
                            signal_row.append(e_time)
                            signal_row.append(cont_btc)
                            signal_row.append(cont_eth)
                            signal_row.append(k_price_btc)
                            signal_row.append(k_price_eth)
                            signal_row.append(p_price_btc)
                            signal_row.append(p_price_eth)
                            signal_row.append(value_start)
                            signal_row.append(value_end)
                            signal_row.append(ret)
                            signal_row.append((max(high_value) / value_start) - 1)
                            signal_row.append(len(hold_value))
                            signal_row.append(position)
                            signal_lst.append(signal_row)
                            position = 1

                            s_time = _row.date_time
                            value_start = value
                            k_price_eth = _row.close_e * (1 - fee)
                            k_price_btc = _row.close_b * (1 + fee)
                            cont_btc = int(value * _row.close_b / 2)
                            cont_eth = -int(value * 1000000 / _row.close_e / 2)
                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (_row.close_e - k_price_eth) / 1000000
                            value = value_b + value_e
                            hold_value = []
                            high_value = []
                            hold_value.append(value_start)
                            high_value.append(value_start)
                        else:
                            value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / _row.close_b)
                            value_e = value_e + cont_eth * (_row.close_e - _row.close_e_1) / 1000000
                            value = value_b + value_e

                            if (_row.tickid + 60 + 28800) % (n * 60) == 0:
                                hold_value.append(value)
                            high_value.append(value)
                            if value <= max(hold_value) * (1-loss_stop):
                                e_time = _row.date_time
                                if cont_btc > 0:
                                    p_price_btc = _row.close_b * (1 - fee)
                                    p_price_eth = _row.close_e * (1 + fee)
                                else:
                                    p_price_btc = _row.close_b * (1 + fee)
                                    p_price_eth = _row.close_e * (1 - fee)
                                value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / p_price_btc)
                                value_e = value_e + cont_eth * (p_price_eth - _row.close_e_1) / 1000000
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
                                signal_row.append(value_start)
                                signal_row.append(value_end)
                                signal_row.append(ret)
                                signal_row.append((max(high_value) / value_start) - 1)
                                signal_row.append(len(hold_value))
                                signal_row.append(position)
                                signal_lst.append(signal_row)
                                position = 0
                    pos_lst.append(position)
                    value_lst.append(value)
                    value_lst_b.append(value_b)
                    value_lst_e.append(value_e)
                if len([i for i in value_lst if i < 0]) > 0:
                    continue
                value_df = pd.DataFrame({'value': value_lst[1:],
                                         'close_b': group_.close_b.tolist(),
                                         'date_time': group_.date_time.tolist()})
                signal_state_all = pd.DataFrame(
                    signal_lst, columns=['s_time', 'e_time', 'cont_btc', 'cont_eth', 'k_price_btc',
                                         'k_price_eth', 'p_price_btc', 'p_price_eth', 'value_s',
                                         'value_e', 'ret', 'max_ret', 'hold_day', 'position'])
                # signal_state_all.to_csv('cl_trend/data/signal_neu_strategy.csv')
                for (s_date, e_date) in date_lst:
                    value_df_ = value_df[(value_df['date_time'] >= s_date) & (value_df['date_time'] <= e_date)] \
                        .reset_index(drop=True)
                    back_stime = value_df_.at[0, 'date_time']
                    back_etime = value_df_.at[len(value_df_) - 1, 'date_time']
                    value_lst = value_df_.value.tolist()
                    net_lst = [i / value_lst[0] for i in value_lst]
                    # net_lst_b = [i / value_lst_b[0] for i in value_lst_b]
                    # net_lst_e = [i / value_lst_e[0] for i in value_lst_e]
                    # net_df = pd.DataFrame({'value': value_lst,
                    #                        'net': net_lst,
                    #                        'date_time': value_df_.date_time.tolist(),
                    #                        'close': value_df_.close.tolist()}).assign(
                    #     close=lambda df: df.close / df.close.tolist()[0])
                    # net_df['date_time'] = pd.to_datetime(net_df['date_time'])
                    # net_df.to_csv('cl_trend/data/net_' + str(n) + '.csv')
                    # net_df[['date_time', 'net', 'close']].plot(
                    #     x='date_time', kind='line', grid=True,
                    #     title=period)
                    # plt.show()
                    signal_state = signal_state_all[
                        (signal_state_all['s_time'] >= s_date) & (signal_state_all['e_time'] <= e_date)]
                    win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                    win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                    total_ret = net_lst[-1]
                    sharp, max_retrace, ann_ROR = sharp_maxretrace_ann(net_lst, n)
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
                    state_row.append(win_R_3)
                    state_row.append(win_R_5)
                    state_row.append(ave_max)
                    state_row.append(beta_period)
                    state_row.append(ma_period)
                    state_row.append(loss_stop)
                    state_row.append(back_stime)
                    state_row.append(back_etime)
                    state_lst.append(state_row)
                print('time_spend = ', time.time()-t0)

    res = pd.DataFrame(state_lst,
                       columns=['period', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                'trade_times', 'ave_r', 'ave_hold_days', 'win_r_3', 'win_r_5', 'ave_max',
                                'beta_period', 'ma_period', 'loss_stop', 'back_stime', 'back_etime'])
    res.to_csv('cl_trend/data/neu_strategy_fp_' + str(n) + '.csv')
