# -*- coding: utf-8 -*-
"""
Created on Sun Jul 08 11:01:46 2018

@author: Administrator
"""
from __future__ import division
import numpy as np
import pandas as pd
import os
import talib as tb
import time
import datetime
import math
import matplotlib.pyplot as plt


def get_winR_odds(ret_lst):
    win_lst = [i for i in ret_lst if i > 0]
    loss_lst = [i for i in ret_lst if i < 0]
    win_R = 0
    odds = 1
    ave = 0
    mid_ret = 0
    if len(win_lst) + len(loss_lst) > 0:
        win_R = len(win_lst) / (len(win_lst) + len(loss_lst))
        ave = (sum(win_lst) + sum(loss_lst)) / (len(win_lst) + len(loss_lst))
        odds = 10
        if len(win_lst) == 0:
            win_lst = [0]
        if len(loss_lst) > 0:
            odds = - np.mean(win_lst) / np.mean(loss_lst)
        win_lst.extend(loss_lst)
        mid_ret = np.percentile(win_lst, 50)
    return win_R, odds, ave, mid_ret


def get_winR_max(ret_lst):
    win_lst_3 = [i for i in ret_lst if i > 0.03]
    loss_lst_3 = [i for i in ret_lst if i < 0.03]
    win_lst_5 = [i for i in ret_lst if i > 0.05]
    loss_lst_5 = [i for i in ret_lst if i < 0.05]
    win_R_3 = 0
    win_R_5 = 0
    ave_max = 0
    if len(win_lst_3) + len(loss_lst_3) > 0:
        win_R_3 = len(win_lst_3) / (len(win_lst_3) + len(loss_lst_3))
        ave_max = (sum(win_lst_3) + sum(loss_lst_3)) / (len(win_lst_3) + len(loss_lst_3))
    if len(win_lst_5) + len(loss_lst_5) > 0:
        win_R_5 = len(win_lst_5) / (len(win_lst_5) + len(loss_lst_5))

    return win_R_3, win_R_5, ave_max


def maxRetrace(list):
    '''
    :param list:netlist
    :return: 最大历史回撤
    '''
    row = []
    for i in range(len(list)):
        row.append(1 - list[i] / max(list[:i + 1]))
    Max = max(row)
    return Max


def annROR(netlist, n):
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


def yearsharpRatio(netlist, n):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(365 * 1440 / n, 0.5)


if __name__ == '__main__':
    os.getcwd()
    #    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    # os.chdir(r'E:\mywork\DC')
    t0 = time.time()
    '''
     N1_lst = [30, 60]
    ATR_n_lst = [2, 2.5]
    # N3_lst = [25, 35, 45, 60, 80, 120]
    P_ATR_LST = [11, 15]
    road_period_lst = [10, 20]
    win_stop_lst = [0.2, 0.1, 0.15]
    '''
    N1_lst = [5]  # 短周期周期1
    N2_lst = [20]  # 长周期周期2
    P_ATR_LST = [5]  # ATR周期
    K1_lst = [s / 100 for s in range(20, 90, 5)]  # 近期比例
    K2_lst = [s / 100 for s in range(80, 90, 5)]  # 长期比例
    ATR_n_lst = [1]  # ATR倍数
    s_date = '2017-08-01'
    e_date = '2018-06-30'
    n = 15
    group = pd.read_csv('huobiservice/data/' + 'ethbtc' + '_' + str(n) + 'min.csv').reset_index()
    print(group)
    #    group = group[(group['date_time'] >= s_date) & (group['date_time'] <= e_date)]
    group.loc[:, ['high', 'low', 'close', 'open', 'vol']] = group.loc[:, ['high', 'low', 'close', 'open', 'vol']] \
        .astype(float)
    print(group)
    df_lst = []
    state_lst = []
    for N1 in N1_lst:
        for N2 in N2_lst:
            for ATR_n in ATR_n_lst:
                for K1 in K1_lst:
                    for N_ATR in P_ATR_LST:
                        for K2 in K2_lst:

                            method = 'OCK_' + str(N1) + '_' + str(N2) + '_' + str(K1) + '_' + str(K2) + '_' + str(
                                N_ATR) + '_' + str(ATR_n)
                            signal_lst = []
                            trad_times = 0

                            if len(group) > N1:
                                net = 1
                                net_lst = []
                                group = group.assign(HH_s=lambda df: tb.MAX(df.high.values, N1)) \
                                    .assign(LL_s=lambda df: tb.MIN(df.low.values, N1)) \
                                    .assign(HH_l=lambda df: tb.MAX(df.high.values, N2)) \
                                    .assign(LL_l=lambda df: tb.MIN(df.low.values, N2)) \
                                    .assign(
                                    short_ratio=lambda df: ((df.close - df.LL_s) / (df.HH_s - df.LL_s) - 0.5) / 0.5) \
                                    .assign(
                                    long_ratio=lambda df: ((df.close - df.LL_l) / (df.HH_l - df.LL_l) - 0.5) / 0.5) \
                                    .assign(ma=lambda df: tb.MA(df.close.values, N2)) \
                                    .assign(
                                    atr=lambda df: tb.ATR(df.high.values, df.low.values, df.close.values, N_ATR))

                                group = group.assign(close_1=lambda df: df.close.shift(1)) \
                                    .assign(short_ratio_1=lambda df: df.short_ratio.shift(1)) \
                                    .assign(long_ratio_1=lambda df: df.long_ratio.shift(1)) \
                                    .assign(ma_1=lambda df: df.ma.shift(1)) \
                                    .assign(atr=lambda df: df.atr.shift(1)) \
                                    .assign(c_ma=lambda df: df.close_1 - df.ma_1) \
                                    .assign(ma_zd=lambda df: df.ma_1 - df.ma_1.shift(1))
                                position = 0
                                high_price_pre = 1000000
                                # signal_row = []
                                # stock_row = []
                                for idx, _row in group.iterrows():

                                    if (position == 0) & (_row.long_ratio_1 > K2) & (_row.short_ratio_1 < -K1):
                                        position = 1
                                        s_time = _row.date_time
                                        cost = _row.open
                                        hold_price = []
                                        high_price = []
                                        hold_price.append(_row.open)
                                        high_price.append(cost)
                                        net = net * _row.close / cost
                                    elif (position == 0) & (_row.high > high_price_pre):
                                        position = 1
                                        s_time = _row.date_time
                                        cost = high_price_pre
                                        hold_price = []
                                        high_price = []
                                        hold_price.append(_row.open)
                                        high_price.append(cost)
                                        net = net * _row.close / cost

                                    elif position == 1:
                                        if _row.low < max(hold_price) - _row.atr * ATR_n:
                                            position = 0
                                            trad_times += 1
                                            high_price.append(_row.high)
                                            e_time = _row.date_time
                                            s_price = max(hold_price) - _row.atr * ATR_n
                                            high_price_pre = max(high_price)
                                            ret = s_price / cost - 1
                                            signal_row = []
                                            signal_row.append(s_time)
                                            signal_row.append(e_time)
                                            signal_row.append(cost)
                                            signal_row.append(s_price)
                                            signal_row.append(ret - 0.004)
                                            signal_row.append(max(high_price) / cost - 1)
                                            signal_row.append(len(hold_price))
                                            net = net * (1 + ret) * 0.996
                                            signal_lst.append(signal_row)
                                        else:
                                            high_price.append(_row.high)
                                            hold_price.append(_row.close)
                                            net = net * _row.close / _row.close_1

                                    net_lst.append(net)
                                ann_ROR = annROR(net_lst, n)
                                total_ret = net_lst[-1]
                                max_retrace = maxRetrace(net_lst)
                                sharp = yearsharpRatio(net_lst, n)
                                net_df = pd.DataFrame(net_lst, columns=['net'])
                                # net_df.to_csv('cl_trend/data/net_' + method + '.csv')

                                signal_state = pd.DataFrame(signal_lst,
                                                            columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                                     'max_ret', 'hold_day']) \
                                    .assign(method=method)
                                df_lst.append(signal_state)
                                win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                                win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                                state_row = []
                                state_row.append(win_r)
                                state_row.append(odds)
                                state_row.append(total_ret - 1)
                                state_row.append(ann_ROR)
                                state_row.append(sharp)
                                state_row.append(max_retrace)
                                state_row.append(len(signal_state))
                                state_row.append(ave_r)
                                state_row.append(signal_state.hold_day.mean())
                                state_row.append(mid_r)
                                state_row.append(win_R_3)
                                state_row.append(win_R_5)
                                state_row.append(ave_max)
                                state_row.append(method)
                                state_lst.append(state_row)
                                #                              print(u'回测开始日期= %s'%s_date)
                                #                              print(u'回测结束日期= %s'%e_date)
                                #                              print(u'胜率= %f'%win_r)
                                #                              print(u'盈亏比= %f '%odds)
                                #                              print(u'总收益= %f'%total_ret)
                                #                              print(u'年化收益= %f'%ann_ROR)
                                #                              print(u'夏普比率= %f' %sharp)
                                #                              print(u'最大回撤=%f'%max_retrace)
                                #                              print(u'交易次数=%d'%len(signal_state))
                                #                              print(u'平均每次收益=%f'%ave_r)
                                #                              print(u'平均持仓周期=%d'%signal_state.hold_day.mean())
                                #                              print(u'中位数收益=%f'%mid_r)
                                #                              print(u'超过3%胜率={}'.format(win_R_3))
                                #                              print(u'超过5%胜率={}'.format(win_R_5))
                                #                              print(u'平均最大收益=%f'%ave_max)
                                #                              print(u'参数=%s'%method)
                                # os.chdir(r'E:\mywork\DC\result')
                                plt.figure()
                                plt.plot(net_lst)
                                plt.title(method)
                                plt.show()
                                # plt.savefig(method + str(n) + '.jpg')
    signal_state = pd.DataFrame(state_lst, columns=['win_r', 'odds', 'total_ret', 'ann_ret', 'sharp', 'max_retrace',
                                                    'trade_times', 'ave_r', 'ave_hold_days', 'mid_r', 'win_r_3',
                                                    'win_r_5', 'ave_max', 'method'])
    # signal_state.to_csv('state_tqa.csv')
    signal_df = pd.concat(df_lst)
    # signal_df.to_csv('signal_tqa.csv')
