# coding=utf-8
'''
Created on 9.30, 2018
适用于eos/btc，ubtc计价并结算
@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import talib as tb
import matplotlib.pyplot as plt
from matplotlib import style
from dataapi import get_exsymbol_kline

style.use('ggplot')

if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 8
    symble_lst = ['ethbtc', 'eosbtc', 'etcbtc', 'iotabtc', 'iostbtc', 'ltcbtc', 'neobtc', 'trxbtc', 'xrpbtc', 'xlmbtc',
                'adabtc', 'ontbtc', 'bnbbtc', 'bchabcbtc', 'bchsvbtc', 'htbtc']
    n = 240  # 回测周期
    period = '240m'
    road_period_l_lst = [i for i in range(22, 23, 1)]  # 通道周期
    K1_lst = [i/100 for i in range(40, 41, 5)]  # 浮动倍数
    N1_lst = [10]  # 均线周期
    P_ATR_LST = [18]  # ATR周期
    ATR_n_lst = [0.9]  # ATR倍数
    status_days_lst = [i for i in range(201)]
    lever = 1  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever
    fee = 0.002
    date_lst = [('2017-10-01', '2018-01-01'), ('2018-01-01', '2018-07-01'), ('2018-07-01', '2019-12-01')]
    # date_lst = [('2017-01-01', '2019-01-02')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        group = pd.read_csv('data/' + symble + '_' + str(n) + 'm.csv') \
            .assign(close_1=lambda df: df.close.shift(1))
        group_day = pd.read_csv('data/' + symble + '_1440m.csv')
        for N_ATR in P_ATR_LST:
            if len(group_day) == 0:
                continue
            group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                         group_day['close'].values, N_ATR)
            for N1 in N1_lst:
                group_day['ma'] = talib.MA(group_day['close'].shift(1).values, N1)
                day_atr = group_day[['date_time', 'atr', 'ma']] \
                    .assign(atr=lambda df: df.atr.shift(1)) \
                    .merge(group, on=['date_time'], how='right') \
                    .assign(c_ma=lambda df: df.close_1 - df.ma) \
                    .assign(ma_zd=lambda df: df['ma'] - df['ma'].shift(1)) \
                    .sort_values(['date_time']).fillna(method='ffill') \
                    .reset_index(drop=True)

                for road_period_l in road_period_l_lst:
                    if len(day_atr) <= road_period_l:
                        continue
                    day_atr = day_atr.assign(HH_LC_l=lambda df: tb.MAX(df.high.values, road_period_l) - tb.MIN(df.close.values, road_period_l)) \
                        .assign(HC_LL_l=lambda df: tb.MAX(df.close.values, road_period_l) - tb.MIN(df.low.values, road_period_l)) \
                        .assign(max_HCL_l=lambda df: [max(row.HH_LC_l, row.HC_LL_l) for idx, row in df.iterrows()])
                    for K1 in K1_lst:
                        group__ = day_atr.assign(buyRange_1=lambda df: K1 * df.max_HCL_l.shift(1) + df.open)
                        for (s_date, e_date) in date_lst:
                            group_ = group__[
                                (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)].dropna() \
                                .reset_index(drop=True)
                            if len(group_) < 20:
                                continue
                            back_stime = group_.at[0, 'date_time']
                            back_etime = group_.at[len(group_) - 1, 'date_time']
                            for ATR_n in ATR_n_lst:
                                for status_day in status_days_lst:
                                    print(symble)
                                    signal_lst = []
                                    trad_times = 0
                                    net = 1
                                    net_lst = []
                                    pos_lst = []
                                    pos = 0
                                    low_price_pre = 0
                                    high_price_pre = 100000000
                                    status = -10000
                                    for idx, _row in group_.iterrows():
                                        if pos == 0:
                                            if (status >= 0) & (status < status_day):
                                                status += 1
                                                if _row.high > high_price_pre:
                                                    s_time = _row.date_time
                                                    cost = high_price_pre * (1 + fee)
                                                    if max(_row.open, _row.low) > high_price_pre:
                                                        cost = max(_row.open, _row.low) * (1 + fee)
                                                    pos = 1
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost)
                                                    high_price.append(cost)
                                                    net = (pos * _row.close / cost + (1 - pos)) * net
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
                                                    net = ((1 + pos) - pos * (2 - _row.close / cost)) * net
                                                else:
                                                    net = net
                                            else:
                                                if (_row.high > _row.buyRange_1) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                                    cost = _row.buyRange_1 * (1 + fee)
                                                    pos = 1
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost)
                                                    high_price.append(cost)
                                                    net = (pos * _row.close / cost + (1 - pos)) * net
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
                                                    net = (pos * _row.close / cost + (1 - pos)) * net
                                                else:
                                                    net = net
                                        else:
                                            if pos > 0:
                                                if _row.low < max(hold_price) - _row.atr * ATR_n:

                                                    trad_times += 1
                                                    high_price.append(_row.high)
                                                    e_time = _row.date_time
                                                    s_price = (max(hold_price) - _row.atr * ATR_n) * (1 - fee)
                                                    if min(_row.open, _row.high) < max(
                                                            hold_price) - _row.atr * ATR_n:
                                                        s_price = min(_row.open, _row.high) * (1 - fee)
                                                    net1 = (pos * s_price / _row.close_1 + (1 - pos))
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
                                                    status = 0
                                                    net = net1 * net
                                                else:
                                                    high_price.append(_row.high)
                                                    hold_price.append(_row.close)
                                                    net = net * (pos * _row.close / _row.close_1 + (1 - pos))

                                        net_lst.append(net)
                                        pos_lst.append(pos)
                                    net_df = pd.DataFrame({'net': net_lst,
                                                           'date_time': group_.date_time.tolist(),
                                                           'close': group_.close.tolist(),
                                                           'pos': pos_lst}).assign(
                                        close=lambda df: df.close / df.close.tolist()[0])
                                    # net_df.to_csv('data/dualtrust_btcindex_' + period + '.csv')
                                    # net_df[['date_time', 'net', 'close']].plot(
                                    #     x='date_time', kind='line', grid=True,
                                    #     title=symble + '_' + period)
                                    #
                                    # plt.show()
                                    ann_ROR = annROR(net_lst, n)
                                    total_ret = net_lst[-1]
                                    max_retrace = maxRetrace(net_lst, n)
                                    sharp = yearsharpRatio(net_lst, n)

                                    signal_state = pd.DataFrame(
                                        signal_lst, columns=['s_time', 'e_time', 'k_price', 'p_price', 'ret',
                                                             'max_ret', 'hold_day', 'position', 'bspk'])
                                    # signal_state.to_csv('cl_trend/data/signal_DT_' + str(n) + '_' + symble + '_' + back_stime + 'ls.csv')
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
                                    state_row.append(K1)
                                    state_row.append(N1)
                                    state_row.append(status_day)
                                    state_row.append(back_stime)
                                    state_row.append(back_etime)
                                    state_lst.append(state_row)
    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days', 'win_r_3',
                                                    'win_r_5', 'ave_max', 'art_N', 'art_n', 'road_period_l',
                                                    'K1', 'N1', 'status_day', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_dt_btc' + str(n) + '.csv')

