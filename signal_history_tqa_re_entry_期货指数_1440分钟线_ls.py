# coding=utf-8
'''
Created on 9.30, 2018
适用于eos/btc，ubtc计价并结算
@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
from dataapi import get_exsymbol_kline
import statsmodels.api as sm
import talib as tb
style.use('ggplot')


def cal_k(low_lst, high_lst, period_k):
    k_lst = []
    r_lst = []

    for i in range(len(low_lst)):
        if i < period_k:
            k_lst.append(0)
            r_lst.append(0)
        else:
            result = (sm.OLS(high_lst[i-period_k:i], sm.add_constant(low_lst[i-period_k:i]))).fit()
            k_lst.append(result.params[1])
            r_lst.append(result.rsquared)
    return k_lst, r_lst


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 8
    symble_lst = ['btcusdt', 'ethusdt', 'eosbtc']
    n = 1440  # 回测周期
    period = '1440m'

    road_period_l_lst = [i for i in range(4, 11, 1)]  # 唐奇安通道周期
    road_period_s_lst = [i for i in range(4, 11, 1)]  # 唐奇安通道周期
    ma_period_lst = [20, 15, 10]
    P_ATR_LST = [14, 16, 20, 12, 10]  # ATR周期
    ATR_n_lst = [0.5, 0.6, 0.7, 0.4]  # ATR倍数

    lever = 1  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever
    fee = 0.0036
    date_lst = [('2017-01-01', '2017-10-01'), ('2017-10-01', '2018-01-01'), ('2018-01-01', '2018-11-01')]
    # date_lst = [('2017-01-01', '2018-10-01')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        if symble == 'ethusd':
            group = pd.read_csv('data/ethusd_' + str(n) + 'm.csv') \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/ethusd_1440m.csv') \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x) + ' 00:00:00'))
        elif symble == 'eosbtc':
            group = pd.read_csv('data/eosbtc_' + str(n) + 'm.csv') \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/eosbtc_1440m.csv')
        else:
            group = pd.read_csv('data/btc_index_' + period + '.csv') \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))\
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/btc_index_' + '1440m' + '.csv')

            for N_ATR in P_ATR_LST:
                if len(group_day) < N_ATR:
                    continue

                group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                             group_day['close'].values, N_ATR)
                for ma_period in ma_period_lst:
                    if len(group_day) <= ma_period:
                        continue
                    group_day['ma'] = talib.MA(group_day['close'].shift(1).values, ma_period)
                    day_atr = group_day[['date_time', 'atr', 'ma']] \
                        .assign(ma_zd=lambda df: df['ma'] - df['ma'].shift(1))\
                        .assign(atr=lambda df: df.atr.shift(1)) \
                        .merge(group, on=['date_time'], how='right') \
                        .sort_values(['date_time']).fillna(method='ffill')\
                        .reset_index(drop=True)
                    print(day_atr)
                    back_stime = day_atr.at[0, 'date_time']
                    back_etime = day_atr.at[len(day_atr) - 1, 'date_time']

                    for road_period_l in road_period_l_lst:
                        for road_period_s in road_period_s_lst:
                            if len(day_atr) <= min(road_period_l, road_period_s):
                                continue
                            day_atr['h10'] = talib.MAX(day_atr['high'].shift(1).values, road_period_l)
                            day_atr['l10'] = talib.MIN(day_atr['low'].shift(1).values, road_period_s)
                            group_ = day_atr\
                                .assign(c_ma=lambda df: df.close_1 - df.ma) \
                                .dropna().reset_index(drop=True)
                            # group_.to_csv('cl_trend/data/atr_' + symble + '.csv')
                            for ATR_n in ATR_n_lst:

                                print(symble)
                                signal_lst = []
                                trad_times = 0
                                net = 1
                                net_lst = []

                                # group_.to_csv('cl_trend/day_atr' + symble + '.csv')\

                                if (symble == 'ethusdt') | (symble == 'eosbtc'):
                                    pos_lst = []

                                    pos = 0

                                    low_price_pre = 0
                                    high_price_pre = 100000000

                                    for idx, _row in group_.iterrows():
                                        if pos == 0:

                                            if (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                                cost = _row.h10 * (1 + fee)
                                                if _row.open > _row.h10:
                                                    cost = _row.open * (1 + fee)
                                                pos = 1

                                                s_time = _row.date_time
                                                hold_price = []
                                                high_price = []
                                                hold_price.append(cost)
                                                high_price.append(cost)
                                                net = _row.close / cost * net

                                            elif (_row.low < _row.l10) & (_row.c_ma < 0) & (_row.ma_zd < 0):

                                                cost = _row.l10 * (1 - fee)
                                                if _row.open < _row.l10:
                                                    cost = _row.open * (1 - fee)

                                                pos = -1
                                                s_time = _row.date_time
                                                hold_price = []
                                                low_price = []
                                                hold_price.append(cost)
                                                low_price.append(cost)
                                                net = (2 - _row.close / cost) * net
                                            elif _row.high > high_price_pre:
                                                s_time = _row.date_time
                                                cost = high_price_pre * (1 + fee)
                                                if max(_row.open, _row.low) > high_price_pre:
                                                    cost = max(_row.open, _row.low) * (1 + fee)
                                                pos = 1
                                                hold_price = []
                                                high_price = []
                                                hold_price.append(cost)
                                                high_price.append(cost)
                                                net = (_row.close / cost) * net
                                            elif _row.low < low_price_pre:
                                                cost = low_price_pre * (1 - fee)
                                                if min(_row.open, _row.high) < low_price_pre:
                                                    cost = min(_row.open, _row.high) * (1 - fee)
                                                pos = -1
                                                s_time = _row.date_time
                                                hold_price = []
                                                low_price = []
                                                hold_price.append(cost)
                                                low_price.append(cost)
                                                net = ((cost - _row.close) / cost + 1) * net
                                            else:
                                                net = net
                                        else:
                                            if pos == 1:
                                                if (_row.low < _row.l10) & (_row.c_ma < 0) & (_row.ma_zd < 0):

                                                    s_price = _row.l10 * (1 - fee)
                                                    if _row.open < _row.l10:
                                                        s_price = _row.open * (1 - fee)
                                                    trad_times += 1
                                                    net1 = s_price / _row.close_1
                                                    ret = s_price / cost - 1
                                                    e_time = _row.date_time
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(s_price)
                                                    signal_row.append(ret)
                                                    signal_row.append(
                                                        (max(high_price) / cost) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('spk')
                                                    signal_lst.append(signal_row)
                                                    s_time = _row.date_time
                                                    cost = _row.l10 * (1 - fee)
                                                    if _row.open < _row.l10:
                                                        cost = _row.open * (1 - fee)

                                                    pos = -1
                                                    net2 = (cost - _row.close) / cost + 1
                                                    hold_price = []
                                                    low_price = []
                                                    hold_price.append(cost)
                                                    low_price.append(cost)
                                                    net = net1 * net2 * net
                                                elif _row.low < max(hold_price) - _row.atr * ATR_n:

                                                    trad_times += 1
                                                    high_price.append(_row.high)
                                                    e_time = _row.date_time
                                                    s_price = (max(hold_price) - _row.atr * ATR_n) * (1 - fee)
                                                    if min(_row.open, _row.high) < max(
                                                            hold_price) - _row.atr * ATR_n:
                                                        s_price = min(_row.open, _row.high) * (1 - fee)
                                                    net1 = s_price / _row.close_1
                                                    ret = s_price / cost - 1
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(s_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((max(high_price) / cost) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('sp')
                                                    signal_lst.append(signal_row)
                                                    pos = 0
                                                    high_price_pre = max(high_price)

                                                    net = net1 * net

                                                else:
                                                    high_price.append(_row.high)
                                                    hold_price.append(_row.close)
                                                    net = net * (_row.close / _row.close_1)

                                            elif pos == -1:
                                                if (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                                    b_price = _row.h10 * (1 + fee)
                                                    if _row.open > _row.h10:
                                                        b_price = _row.open * (1 + fee)

                                                    e_time = _row.date_time
                                                    trad_times += 1
                                                    net1 = (_row.close_1 - b_price) / _row.close_1 + 1
                                                    ret = (cost - b_price) / cost
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(b_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((cost - min(low_price)) / cost + 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('bpk')
                                                    signal_lst.append(signal_row)
                                                    pos = 1

                                                    cost = _row.h10 * (1 + fee)
                                                    if _row.open > _row.h10:
                                                        cost = _row.open * (1 + fee)

                                                    net2 = _row.close / cost
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost)
                                                    high_price.append(cost)
                                                    net = net1 * net2 * net

                                                elif _row.high > min(hold_price) + _row.atr * ATR_n:
                                                    trad_times += 1
                                                    e_time = _row.date_time
                                                    b_price = (min(hold_price) + _row.atr * ATR_n) * (1 + fee)
                                                    if max(_row.open, _row.low) > min(
                                                            hold_price) + _row.atr * ATR_n:
                                                        b_price = max(_row.open, _row.low) * (1 + fee)
                                                    net1 = (_row.close_1 - b_price) / _row.close_1 + 1
                                                    ret = (cost - b_price) / cost
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(b_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((cost - min(low_price)) / cost + 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('bp')
                                                    signal_lst.append(signal_row)
                                                    pos = 0
                                                    low_price.append(_row.low)

                                                    net = net * net1
                                                    low_price_pre = min(low_price)

                                                else:
                                                    low_price.append(_row.low)
                                                    hold_price.append(_row.close)
                                                    net = net * ((_row.close_1 - _row.close) / _row.close_1 + 1)
                                        net_lst.append(net)
                                        pos_lst.append(pos)
                                    # net_df = pd.DataFrame({'net': net_lst,
                                    #                        'date_time': group_.date_time.tolist(),
                                    #                        'close': group_.close.tolist(),
                                    #                        'pos': pos_lst}).assign(
                                    #     close=lambda df: df.close / df.close.tolist()[0])
                                    # net_df.to_csv('data/net_.csv')
                                    # net_df[['date_time', 'net', 'close']].plot(
                                    #     x='date_time', kind='line', grid=True,
                                    #     title=symble + '_' + period)
                                    #
                                    # plt.show()
                                    ann_ROR = annROR(net_lst, n)
                                    total_ret = net_lst[-1]
                                    max_retrace = maxRetrace(net_lst)
                                    sharp = yearsharpRatio(net_lst, n)

                                    signal_state = pd.DataFrame(
                                        signal_lst, columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                             'max_ret', 'hold_day', 'position', 'bspk'])

                                    # signal_state.to_csv('cl_trend/data/signal_gzw_' + str(n) + '_' + symble + '_' + back_stime + 'ls.csv')
                                    # df_lst.append(signal_state)
                                    win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                                    win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                                    state_row = []
                                    state_row.append(symble)
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

                                    state_row.append(N_ATR)
                                    state_row.append(ATR_n)
                                    state_row.append(road_period_l)
                                    state_row.append(road_period_s)
                                    state_row.append(ma_period)
                                    state_row.append(back_stime)
                                    state_row.append(back_etime)
                                    state_lst.append(state_row)
                                elif symble == 'btcusdt':
                                    pos_lst = []
                                    low_price_pre = 0
                                    high_price_pre = 100000000
                                    pos = 1
                                    for idx, _row in group_.iterrows():
                                        if pos == 1:

                                            if (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):

                                                cost = _row.h10 * (1 + fee)
                                                if _row.open > _row.h10:
                                                    cost = _row.open * (1 + fee)

                                                net1 = (pos * cost + (1 - pos) * _row.close_1) / cost
                                                pos = 2
                                                net2 = (pos * _row.close + (1 - pos) * cost) / _row.close
                                                s_time = _row.date_time
                                                hold_price = []
                                                high_price = []
                                                hold_price.append(cost)
                                                high_price.append(cost)
                                                net = net1 * net2 * net
                                            elif (_row.low < _row.l10) & (_row.c_ma < 0) & (_row.ma_zd < 0):

                                                cost = _row.l10 * (1 - fee)

                                                if _row.open < _row.l10:
                                                    cost = _row.open * (1 - fee)

                                                net1 = (pos * cost + (1 - pos) * _row.close_1) / cost
                                                pos = 0
                                                net2 = (pos * _row.close + (1 - pos) * cost) / _row.close
                                                s_time = _row.date_time
                                                hold_price = []
                                                low_price = []
                                                hold_price.append(cost)
                                                low_price.append(cost)
                                                net = net1 * net2 * net
                                            elif _row.high > high_price_pre:
                                                s_time = _row.date_time
                                                cost = high_price_pre * (1 + fee)
                                                if max(_row.open, _row.low) > high_price_pre:
                                                    cost = max(_row.open, _row.low) * (1 + fee)
                                                net1 = (pos * cost + (
                                                        1 - pos) * _row.close_1) / cost
                                                pos = 2
                                                net2 = (pos * _row.close + (
                                                        1 - pos) * cost) / _row.close
                                                hold_price = []
                                                high_price = []
                                                hold_price.append(cost)
                                                high_price.append(cost)
                                                net = net1 * net2 * net
                                            elif _row.low < low_price_pre:
                                                cost = low_price_pre * (1 - fee)
                                                if min(_row.open, _row.high) < low_price_pre:
                                                    cost = min(_row.open, _row.high) * (1 - fee)
                                                net1 = (pos * cost + (
                                                        1 - pos) * _row.close_1) / cost
                                                pos = 0
                                                net2 = (pos * _row.close + (
                                                        1 - pos) * cost) / _row.close
                                                s_time = _row.date_time
                                                hold_price = []
                                                low_price = []
                                                hold_price.append(cost)
                                                low_price.append(cost)
                                                net = net1 * net2 * net
                                            else:
                                                net = net
                                        else:
                                            if pos > 1:
                                                if (_row.low < _row.l10) & (_row.c_ma < 0) & (_row.ma_zd < 0):

                                                    s_price = _row.l10 * (1 - fee)
                                                    if _row.open < _row.l10:
                                                        s_price = _row.open * (1 - fee)

                                                    trad_times += 1
                                                    net1 = (pos * s_price + (
                                                                1 - pos) * _row.close_1) / s_price
                                                    ret = s_price / cost - 1
                                                    e_time = _row.date_time
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(s_price)
                                                    signal_row.append(ret)
                                                    signal_row.append(
                                                        (pos * max(high_price) + (1 - pos) * cost) / max(
                                                            high_price) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('spk')
                                                    signal_lst.append(signal_row)
                                                    s_time = _row.date_time
                                                    cost = _row.l10 * (1 - fee)
                                                    if _row.open < _row.l10:
                                                        cost = _row.open * (1 - fee)

                                                    pos = 0
                                                    net2 = (pos * _row.close + (
                                                                1 - pos) * cost) / _row.close
                                                    hold_price = []
                                                    low_price = []
                                                    hold_price.append(cost)
                                                    low_price.append(cost)
                                                    net = net1 * net2 * net
                                                elif _row.low < max(hold_price) - _row.atr * ATR_n:

                                                    trad_times += 1
                                                    high_price.append(_row.high)
                                                    e_time = _row.date_time
                                                    s_price = (max(hold_price) - _row.atr * ATR_n) * (
                                                                1 - fee)
                                                    if min(_row.open, _row.high) < max(
                                                            hold_price) - _row.atr * ATR_n:
                                                        s_price = min(_row.open, _row.high) * (1 - fee)
                                                    net1 = (pos * s_price + (
                                                                1 - pos) * _row.close_1) / s_price
                                                    ret = s_price / cost - 1
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(s_price)
                                                    signal_row.append(ret)
                                                    signal_row.append(
                                                        (pos * max(high_price) + (1 - pos) * cost) / max(
                                                            high_price) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('sp')
                                                    signal_lst.append(signal_row)
                                                    pos = 1
                                                    net2 = (pos * _row.close + (
                                                                1 - pos) * s_price) / _row.close
                                                    high_price_pre = max(high_price)

                                                    net = net1 * net2 * net

                                                elif 1 - _row.low / _row.close_1 > 1 / pos:
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(_row.low)
                                                    signal_row.append(-1)
                                                    signal_row.append(
                                                        (pos * max(high_price) + (1 - pos) * cost) / max(
                                                            high_price) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('boom')
                                                    signal_lst.append(signal_row)
                                                    net = 0.000001
                                                    break

                                                else:
                                                    high_price.append(_row.high)
                                                    hold_price.append(_row.close)
                                                    net = net * (pos * _row.close + (
                                                            1 - pos) * _row.close_1) / _row.close
                                                    pos = pos * _row.close / (
                                                            pos * _row.close + (1 - pos) * _row.close_1)
                                            elif pos == 0:
                                                if (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):

                                                    b_price = _row.h10 * (1 + fee)
                                                    if _row.open > _row.h10:
                                                        b_price = _row.open * (1 + fee)

                                                    e_time = _row.date_time
                                                    trad_times += 1
                                                    net1 = (pos * b_price + (
                                                                1 - pos) * _row.close_1) / b_price
                                                    ret = (pos * b_price + (1 - pos) * cost) / b_price - 1
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(b_price)
                                                    signal_row.append(ret)
                                                    signal_row.append(
                                                        (pos * min(low_price) + (1 - pos) * cost) / min(
                                                            low_price) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('bpk')
                                                    signal_lst.append(signal_row)
                                                    pos = 2

                                                    cost = _row.h10 * (1 + fee)
                                                    if _row.open > _row.h10:
                                                        cost = _row.open * (1 + fee)

                                                    net2 = (pos * _row.close + (
                                                                1 - pos) * cost) / _row.close
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost)
                                                    high_price.append(cost)
                                                    net = net1 * net2 * net

                                                elif _row.high > min(hold_price) + _row.atr * ATR_n:
                                                    trad_times += 1
                                                    e_time = _row.date_time
                                                    b_price = (min(hold_price) + _row.atr * ATR_n) * (
                                                                1 + fee)
                                                    if max(_row.open, _row.low) > min(
                                                            hold_price) + _row.atr * ATR_n:
                                                        b_price = max(_row.open, _row.low) * (1 + fee)
                                                    net1 = (pos * b_price + (
                                                                1 - pos) * _row.close_1) / b_price
                                                    ret = (pos * b_price + (1 - pos) * cost) / b_price - 1
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(b_price)
                                                    signal_row.append(ret)
                                                    signal_row.append(
                                                        (pos * min(low_price) + (1 - pos) * cost) / min(
                                                            low_price) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('bp')
                                                    signal_lst.append(signal_row)
                                                    pos = 1
                                                    net2 = (pos * _row.close + (
                                                                1 - pos) * b_price) / _row.close
                                                    low_price.append(_row.low)

                                                    net = net * net1 * net2
                                                    low_price_pre = min(low_price)

                                                else:
                                                    low_price.append(_row.low)
                                                    hold_price.append(_row.close)
                                                    net = net * (pos * _row.close + (
                                                            1 - pos) * _row.close_1) / _row.close
                                        net_lst.append(net)
                                        pos_lst.append(pos)
                                    # net_df = pd.DataFrame({'net': net_lst,
                                    #                        'date_time': group_.date_time.tolist(),
                                    #                        'close': group_.close.tolist(),
                                    #                        'pos': pos_lst}).assign(
                                    #     close=lambda df: df.close / df.close.tolist()[0])
                                    # net_df.to_csv('data/net_.csv')
                                    # net_df[['date_time', 'net', 'close']].plot(
                                    #     x='date_time', kind='line', grid=True,
                                    #     title=symble + '_' + period)
                                    #
                                    # plt.show()
                                    ann_ROR = annROR(net_lst, n)
                                    total_ret = net_lst[-1]
                                    max_retrace = maxRetrace(net_lst)
                                    sharp = yearsharpRatio(net_lst, n)

                                    signal_state = pd.DataFrame(
                                        signal_lst, columns=['s_time', 'e_time', 'b_price', 's_price',
                                                             'ret', 'max_ret', 'hold_day', 'position', 'bspk'])

                                    # signal_state.to_csv('cl_trend/data/signal_gzw_' + str(n) + '_' + symble + '_' + back_stime + 'ls.csv')
                                    # df_lst.append(signal_state)
                                    win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                                    win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                                    state_row = []
                                    state_row.append(symble)
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

                                    state_row.append(N_ATR)
                                    state_row.append(ATR_n)
                                    state_row.append(road_period_l)
                                    state_row.append(road_period_s)
                                    state_row.append(ma_period)

                                    state_row.append(back_stime)
                                    state_row.append(back_etime)
                                    state_lst.append(state_row)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'road_period_l',
                                                    'road_period_s', 'ma_period', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_tqa_' + str(n) + 'ls.csv')

