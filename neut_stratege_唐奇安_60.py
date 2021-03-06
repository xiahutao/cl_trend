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
    period = '60m'
    n = 60
    now = datetime.datetime.now()
    road_period_l_lst = [35]  # 唐奇安通道周期

    ma_period_lst = [10]
    P_ATR_LST = [40]  # ATR周期
    ATR_n_lst = [1.1]  # ATR倍数
    status_days_lst = [0]
    win_stop_lst = [10]

    df_btc = pd.read_csv('data/btcusdt_1m.csv').loc[
             :, ['tickid', 'open', 'close']] \
        .rename(columns={'open': 'open_b', 'close': 'close_b'}).assign(close_b_1=lambda df: df.close_b.shift(1))
    df_eth = pd.read_csv('data/ethusdt_1m.csv').loc[
             :, ['tickid', 'open', 'close']] \
        .rename(columns={'open': 'open_e', 'close': 'close_e'}).assign(close_e_1=lambda df: df.close_e.shift(1))
    df_org = df_btc.merge(df_eth, on=['tickid'])
    df_org = df_org[df_org['tickid'] >= 1493568000]
    del df_btc
    del df_eth
    hq_data = pd.read_csv('data/ethbtc_1m.csv').loc[
              :, ['tickid', 'close', 'high', 'low', 'open']]
    hq_data = hq_data[hq_data['tickid'] >= 1493568000]
    group = pd.read_csv('data/ethbtc_' + period + '.csv').loc[:, ['tickid', 'open', 'high', 'low', 'close']]
    group = group[group['tickid'] >= 1493568000]
    group_day = pd.read_csv('data/ethbtc_' + '1440m' + '.csv').loc[:, ['tickid', 'open', 'high', 'low', 'close']]
    group_day = group_day[group_day['tickid'] >= 1493568000]
    fee = 0.002
    date_lst = [('2017-08-01', '2019-02-30'), ('2017-08-01', '2018-01-01'), ('2018-01-01', '2018-07-01'),
                ('2018-07-01', '2019-02-30')]
    # date_lst = [('2017-01-01', '2019-02-30')]
    state_lst = []
    for N_ATR in P_ATR_LST:
        if len(group_day) < N_ATR:
            continue
        group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                     group_day['close'].values, N_ATR)
        for ma_period in ma_period_lst:
            if len(group_day) <= ma_period:
                continue
            group_day['ma'] = talib.MA(group_day['close'].shift(1).values, ma_period - 1)
            day_atr = group_day[['tickid', 'atr', 'ma', 'close']] \
                .assign(close_n=lambda df: df['close'].shift(ma_period)) \
                .assign(atr=lambda df: df.atr.shift(1)) \
                .merge(group, on=['tickid'], how='right') \
                .sort_values(['tickid']).fillna(method='ffill') \
                .reset_index(drop=True)[['tickid', 'atr', 'ma', 'close_n', 'high', 'low']]
            print(day_atr)
            for road_period_l in road_period_l_lst:
                road_period_s = road_period_l

                if len(day_atr) <= min(road_period_l, road_period_s):
                    continue
                day_atr['h10'] = talib.MAX(day_atr['high'].shift(1).values, road_period_l)
                day_atr['l10'] = talib.MIN(day_atr['low'].shift(1).values, road_period_s)
                group_ = day_atr[['tickid', 'close_n', 'h10', 'l10', 'atr', 'ma']] \
                    .merge(hq_data, on=['tickid'], how='right').sort_values(['tickid']).fillna(method='ffill') \
                    .merge(df_org, on='tickid').sort_values(['tickid'])\
                    .assign(c_ma=lambda df: df.close - df.ma)\
                    .assign(ma_zd=lambda df: df.close - df.close_n) \
                    .assign(date_time=lambda df: df.tickid.apply(lambda x: str(datetime.datetime.fromtimestamp(x)))) \
                    .dropna().reset_index(drop=False)
                # del day_atr
                print(group_)
                # group_.to_csv('cl_trend/data/data.csv')
                for ATR_n in ATR_n_lst:
                    for win_stop in win_stop_lst:
                        for status_day in status_days_lst:
                            print(status_day)
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
                            low_price_pre = 0
                            high_price_pre = 100000000
                            t0 = time.time()
                            for idx, _row in group_.iterrows():
                                if position == 0:
                                    if (status >= 0) & (status < status_day):
                                        status += 1
                                        if _row.close > high_price_pre:
                                            cost = _row.close
                                            s_time = _row.date_time
                                            value_start = value
                                            position = 1
                                            k_price_eth = _row.close_e * (1 + fee)
                                            k_price_btc = _row.close_b * (1 - fee)
                                            cont_btc = -int(value * _row.close_b / 2)
                                            cont_eth = int(value * 1000000 / _row.close_e / 2)
                                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                                            value_e = value_e + cont_eth * (
                                                    _row.close_e - k_price_eth) / 1000000
                                            value = value_b + value_e
                                            hold_value = []
                                            hold_value.append(value_start)
                                            hold_price = []
                                            high_price = []
                                            hold_price.append(cost)
                                            high_price.append(cost)
                                        elif _row.close < low_price_pre:
                                            cost = _row.close
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
                                            hold_value.append(value_start)
                                            hold_price = []
                                            low_price = []
                                            hold_price.append(cost)
                                            low_price.append(cost)
                                        else:
                                            value = value
                                            value_b = value_b
                                            value_e = value_e
                                    else:
                                        if (_row.close > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                            cost = _row.close
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
                                            hold_value.append(value_start)
                                            hold_price = []
                                            high_price = []
                                            hold_price.append(cost)
                                            high_price.append(cost)
                                        elif (_row.close < _row.l10) & (_row.c_ma < 0) & (_row.ma_zd < 0):
                                            cost = _row.close
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
                                            hold_value.append(value_start)
                                            hold_price = []
                                            low_price = []
                                            hold_price.append(cost)
                                            low_price.append(cost)
                                        elif _row.close > high_price_pre:
                                            cost = _row.close
                                            s_time = _row.date_time
                                            value_start = value
                                            position = 1
                                            k_price_eth = _row.close_e * (1 + fee)
                                            k_price_btc = _row.close_b * (1 - fee)
                                            cont_btc = -int(value * _row.close_b / 2)
                                            cont_eth = int(value * 1000000 / _row.close_e / 2)
                                            value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                                            value_e = value_e + cont_eth * (
                                                        _row.close_e - k_price_eth) / 1000000
                                            value = value_b + value_e
                                            hold_value = []
                                            hold_value.append(value_start)
                                            hold_price = []
                                            high_price = []
                                            hold_price.append(cost)
                                            high_price.append(cost)
                                        elif _row.close < low_price_pre:
                                            cost = _row.close
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
                                            hold_value.append(value_start)
                                            hold_price = []
                                            low_price = []
                                            hold_price.append(cost)
                                            low_price.append(cost)
                                        else:
                                            value = value
                                            value_b = value_b
                                            value_e = value_e
                                elif position == 1:
                                    if (_row.close < _row.l10) & (_row.c_ma < 0) & (_row.ma_zd < 0):
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
                                        hold_value.append(value_end)
                                        high_price.append(_row.high)
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
                                        signal_row.append((max(hold_value) / value_start) - 1)
                                        signal_row.append((max(high_price) / cost) - 1)
                                        signal_row.append(len(hold_price))
                                        signal_row.append(position)
                                        signal_lst.append(signal_row)
                                        position = -1
                                        cost = _row.close
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
                                        hold_price = []
                                        low_price = []
                                        hold_price.append(cost)
                                        low_price.append(cost)
                                    elif _row.close < max(hold_price) - _row.atr * ATR_n:
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
                                        hold_value.append(value_end)
                                        high_price.append(_row.high)
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
                                        signal_row.append((max(hold_value) / value_start) - 1)
                                        signal_row.append((max(high_price) / cost) - 1)
                                        signal_row.append(len(hold_price))
                                        signal_row.append(position)
                                        signal_lst.append(signal_row)
                                        position = 0
                                        high_price_pre = max(high_price)
                                    elif _row.close >= cost * (1 + win_stop):
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
                                        hold_value.append(value_end)
                                        high_price.append(_row.high)
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
                                        signal_row.append((max(hold_value) / value_start) - 1)
                                        signal_row.append((max(high_price) / cost) - 1)
                                        signal_row.append(len(hold_price))
                                        signal_row.append(position)
                                        signal_lst.append(signal_row)
                                        position = 0
                                        high_price_pre = max(high_price)
                                    else:
                                        value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / _row.close_b)
                                        value_e = value_e + cont_eth * (_row.close_e - _row.close_e_1) / 1000000
                                        value = value_b + value_e
                                        hold_value.append(value)
                                        if (_row.tickid + 60 + 28800) % (n * 60) == 0:
                                            hold_price.append(_row.close)
                                        high_price.append(_row.high)
                                else:
                                    if (_row.close > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
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
                                        hold_value.append(value_end)
                                        low_price.append(_row.low)
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
                                        signal_row.append((max(hold_value) / value_start) - 1)
                                        signal_row.append((cost - min(low_price)) / cost)
                                        signal_row.append(len(hold_price))
                                        signal_row.append(position)
                                        signal_lst.append(signal_row)
                                        position = 1
                                        cost = _row.close
                                        s_time = _row.date_time
                                        value_start = value
                                        k_price_eth = _row.close_e * (1 + fee)
                                        k_price_btc = _row.close_b * (1 - fee)
                                        cont_btc = -int(value * _row.close_b / 2)
                                        cont_eth = int(value * 1000000 / _row.close_e / 2)
                                        value_b = value_b + cont_btc * (1 / k_price_btc - 1 / _row.close_b)
                                        value_e = value_e + cont_eth * (_row.close_e - k_price_eth) / 1000000
                                        value = value_b + value_e
                                        hold_value = []
                                        high_value = []
                                        hold_value.append(value_start)
                                        high_value.append(value_start)
                                        hold_price = []
                                        high_price = []
                                        hold_price.append(cost)
                                        high_price.append(cost)

                                    elif _row.close > min(hold_price) + _row.atr * ATR_n:
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
                                        hold_value.append(value_end)
                                        low_price.append(_row.low)
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
                                        signal_row.append((max(hold_value) / value_start) - 1)
                                        signal_row.append((cost - min(low_price)) / cost)
                                        signal_row.append(len(hold_price))
                                        signal_row.append(position)
                                        signal_lst.append(signal_row)
                                        low_price_pre = min(low_price)
                                        position = 0
                                    elif _row.close <= cost * (1 - win_stop):
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
                                        hold_value.append(value_end)
                                        low_price.append(_row.low)
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
                                        signal_row.append((max(hold_value) / value_start) - 1)
                                        signal_row.append((cost - min(low_price)) / cost)
                                        signal_row.append(len(hold_price))
                                        signal_row.append(position)
                                        signal_lst.append(signal_row)
                                        low_price_pre = min(low_price)
                                        position = 0

                                    else:
                                        value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / _row.close_b)
                                        value_e = value_e + cont_eth * (_row.close_e - _row.close_e_1) / 1000000
                                        value = value_b + value_e
                                        hold_value.append(value)
                                        if (_row.tickid + 60 + 28800) % (n * 60) == 0:
                                            hold_price.append(_row.close)
                                        low_price.append(_row.low)
                                pos_lst.append(position)
                                value_lst.append(value)
                                value_lst_b.append(value_b)
                                value_lst_e.append(value_e)
                            if len([i for i in value_lst if i < 0]) > 0:
                                continue
                            value_df = pd.DataFrame({'value': value_lst[1:],
                                                     'close': group_.close.tolist(),
                                                     'date_time': group_.date_time.tolist()})
                            signal_state_all = pd.DataFrame(
                                signal_lst, columns=['s_time', 'e_time', 'cont_btc', 'cont_eth', 'k_price_btc',
                                                     'k_price_eth', 'p_price_btc', 'p_price_eth', 'value_s',
                                                     'value_e', 'ret', 'max_ret', 'max_ret_price', 'hold_day',
                                                     'position'])
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
                                win_R_3_p, win_R_5_p, ave_max_p = get_winR_max(signal_state.max_ret_price.tolist())
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
                                state_row.append(win_R_3_p)
                                state_row.append(win_R_5_p)
                                state_row.append(ave_max_p)
                                state_row.append(N_ATR)
                                state_row.append(ATR_n)
                                state_row.append(road_period_l)
                                state_row.append(ma_period)
                                state_row.append(win_stop)
                                state_row.append(status_day)
                                state_row.append(back_stime)
                                state_row.append(back_etime)
                                state_lst.append(state_row)
                            print('time_spend = ', time.time()-t0)

    res = pd.DataFrame(state_lst,
                       columns=['period', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                'trade_times', 'ave_r', 'ave_hold_days', 'win_r_3', 'win_r_5', 'ave_max',
                                'win_r_3_p', 'win_r_5_p', 'ave_max_p', 'art_N', 'art_n', 'road_period_l',
                                'ma_period', 'win_stop', 'status_day', 'back_stime', 'back_etime'])
    res.to_csv('cl_trend/data/neu_strategy_tqa_new_' + str(n) + '.csv')
