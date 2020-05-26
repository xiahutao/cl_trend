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
import talib
import matplotlib.pyplot as plt
import statsmodels.api as sm


def cal_e(btc_lst, eth_lst, period_ols):
    e_lst = []
    for i in range(len(btc_lst)):
        if i < period_ols:
            e_lst.append(0)
        else:
            result = (sm.OLS(eth_lst[i - period_ols:i+1], sm.add_constant(btc_lst[i - period_ols:i+1]))).fit()
            try:
                e_lst.append(eth_lst[i - period_ols:i+1][-1] - result.fittedvalues[-1])
            except Exception as e:
                print(str(e))
                e_lst.append(0)
    return e_lst


if __name__ == '__main__':
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    kind = ['btcusdt', 'ethusdt']
    switch = 1
    period = '240m'
    n = 240
    now = datetime.datetime.now()

    period_std_lst = [i for i in range(33, 34, 1)]  # 标准化窗口
    period_ma_lst = [15]
    period_ols_lst = [100]
    K_lst = [1.3]

    df_btc_period = pd.read_csv('data/btcusdt_' + period + '.csv').loc[
             :, ['tickid', 'close']].assign(close_b_1=lambda df: df.close.shift(1)).rename(columns={'close': 'close_btc'})\
        .assign(tickid=lambda df: df.tickid.apply(lambda x: x+0))
    df_eth_period = pd.read_csv('data/ethusdt_' + period + '.csv').loc[
             :, ['tickid', 'close']].assign(close_e_1=lambda df: df.close.shift(1)).rename(columns={'close': 'close_eth'})\
        .assign(tickid=lambda df: df.tickid.apply(lambda x: x+0))
    group = df_btc_period.merge(df_eth_period, on=['tickid']).sort_values(['tickid'])
    del df_btc_period
    del df_eth_period
    # group = group[group['tickid'] >= 1493568000]

    group_day = pd.read_csv('data/ethbtc_' + '1440m' + '.csv').loc[:, ['tickid', 'close']].sort_values(['tickid'])
    # group_day = group_day[group_day['tickid'] >= 1493568000]
    fee = 0.003
    date_lst = [('2018-01-01', '2019-04-30')]

    # date_lst = [('2017-01-01', '2019-02-30')]
    state_lst = []

    for period_ols in period_ols_lst:
        e_lst = cal_e(group.close_btc.tolist(), group.close_eth.tolist(), period_ols)
        df_signal = pd.DataFrame({'tickid': group.tickid.tolist(), 'e_value': e_lst,
                                  'close_b': group.close_btc.tolist(),
                                  'close_e': group.close_eth.tolist(),
                                  'close_b_1': group.close_b_1.tolist(),
                                  'close_e_1': group.close_e_1.tolist()})

        for period_std in period_std_lst:
            df_ = df_signal.assign(m=lambda df: talib.MA(df.e_value.values, period_std))\
                .assign(d=lambda df: talib.STDDEV(df['e_value'].values, timeperiod=period_std, nbdev=1))\
                .assign(e_v_n=lambda df: (df.e_value - df.m)/df.d)[
                ['tickid', 'e_v_n', 'close_b', 'close_e', 'close_b_1', 'close_e_1']]\
                .sort_values(['tickid'])
            print(df_)
            for period_ma in period_ma_lst:
                group_ = group_day\
                    .assign(c_ma_ethbtc=lambda df: df.close.shift(1) - talib.MA(df.close.shift(1).values, period_ma))\
                    .drop(['close'], axis=1).merge(df_, on=['tickid'], how='right') \
                    .sort_values(['tickid']).fillna(method='ffill') \
                    .assign(date_time=lambda df: df.tickid.apply(lambda x: str(datetime.datetime.fromtimestamp(x)))) \
                    .dropna().reset_index(drop=True)
                print(group_)
                group_[['date_time', 'e_v_n', 'c_ma_ethbtc']].to_csv('cl_trend/data/e_v_n.csv')
                for k in K_lst:

                    signal_lst = []
                    trad_times = 0
                    net_lst = []
                    pos_lst = []
                    net = 1
                    num = 0
                    status = -100000
                    position = 0  # 0:空仓 1：多ETH 空BTC -1：空ETH 多BTC
                    value_b = 10000 / group_.at[0, 'close_b']
                    value_e = 10000 / group_.at[0, 'close_b']
                    value = value_b + value_e
                    value_lst = [value]
                    value_lst_b = [value_b]
                    value_lst_e = [value_e]
                    low_price_pre = 0
                    high_price_pre = 100000000
                    t0 = time.time()
                    for idx, _row in group_.iterrows():
                        if position == 0:

                            if (_row.e_v_n > k) & (_row.c_ma_ethbtc > 0):
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

                            elif (_row.e_v_n < -k) & (_row.c_ma_ethbtc < 0):
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

                            else:
                                value = value
                                value_b = value_b
                                value_e = value_e
                        elif position == 1:
                            if _row.e_v_n < 0:
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
                                signal_row.append(len(hold_value))
                                signal_row.append(position)
                                signal_lst.append(signal_row)
                                position = 0

                            else:
                                value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / _row.close_b)
                                value_e = value_e + cont_eth * (_row.close_e - _row.close_e_1) / 1000000
                                value = value_b + value_e
                                hold_value.append(value)

                        else:
                            if _row.e_v_n > 0:
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
                                signal_row.append(len(hold_value))
                                signal_row.append(position)
                                signal_lst.append(signal_row)
                                position = 0

                            else:
                                value_b = value_b + cont_btc * (1 / _row.close_b_1 - 1 / _row.close_b)
                                value_e = value_e + cont_eth * (_row.close_e - _row.close_e_1) / 1000000
                                value = value_b + value_e
                                hold_value.append(value)

                        pos_lst.append(position)
                        value_lst.append(value)
                        value_lst_b.append(value_b)
                        value_lst_e.append(value_e)
                    if len([i for i in value_lst if i < 0]) > 0:
                        continue
                    value_df = pd.DataFrame({'value': value_lst[1:],
                                             'date_time': group_.date_time.tolist()})
                    signal_state_all = pd.DataFrame(
                        signal_lst, columns=['s_time', 'e_time', 'cont_btc', 'cont_eth', 'k_price_btc',
                                             'k_price_eth', 'p_price_btc', 'p_price_eth', 'value_s',
                                             'value_e', 'ret', 'max_ret', 'hold_day',
                                             'position'])
                    # signal_state_all.to_csv('cl_trend/data/signal_313_' + str(n) + '.csv')
                    for (s_date, e_date) in date_lst:
                        value_df_ = value_df[
                            (value_df['date_time'] >= s_date) & (value_df['date_time'] <= e_date)] \
                            .reset_index(drop=True)
                        back_stime = value_df_.at[0, 'date_time']
                        back_etime = value_df_.at[len(value_df_) - 1, 'date_time']
                        value_lst = value_df_.value.tolist()
                        net_lst = [i / value_lst[0] for i in value_lst]
                        # net_lst_b = [i / value_lst_b[0] for i in value_lst_b]
                        # net_lst_e = [i / value_lst_e[0] for i in value_lst_e]
                        net_df = pd.DataFrame({'net': net_lst,
                                               'date_time': value_df_.date_time.tolist()})
                        # net_df['date_time'] = pd.to_datetime(net_df['date_time'])
                        net_df.to_csv('cl_trend/data/data/313_' + period + '.csv')

                        signal_state = signal_state_all[
                            (signal_state_all['s_time'] >= s_date) & (
                                        signal_state_all['e_time'] <= e_date)]
                        win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                        win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())

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
                        state_row.append(win_R_3)
                        state_row.append(win_R_5)
                        state_row.append(ave_max)

                        state_row.append(period_ols)
                        state_row.append(period_std)
                        state_row.append(period_ma)
                        state_row.append(k)

                        state_row.append(back_stime)
                        state_row.append(back_etime)
                        state_lst.append(state_row)
                    print('time_spend = ', time.time() - t0)

    res = pd.DataFrame(state_lst,
                       columns=['period', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                'trade_times', 'ave_r', 'ave_hold_days', 'win_r_3', 'win_r_5', 'ave_max', 'period_ols',
                                'period_std', 'period_ma', 'k', 'back_stime', 'back_etime'])
    res.to_csv('cl_trend/data/neu_strategy_fp_' + str(n) + '.csv')
