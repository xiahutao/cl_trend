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


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 8
    symble_lst = ["xrpeth", 'ltceth', 'neoeth', 'onteth', 'etceth', 'bnbeth', 'xlmeth', "eoseth", 'trxeth', 'iotaeth',
                  'adaeth', 'iosteth', 'xmreth']
    n = 240  # 回测周期
    period = '240m'

    road_period_l_lst = [i for i in range(18, 21, 1)]  # 唐奇安通道周期
    ma_period_lst = [15]
    P_ATR_LST = [24, 20, 16, 12, 28, 32, 36, 40]  # ATR周期
    ATR_n_lst = [0.6, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # ATR倍数
    # status_days_lst = [0]
    status_days_lst = [i for i in range(0, 1, 3)]
    win_stop_lst = [100000]

    lever = 1  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever
    fee = 0.0036
    date_lst = [('2017-01-01', '2018-01-01'), ('2018-01-01', '2018-07-01'), ('2018-07-01', '2019-05-01')]
    # date_lst = [('2018-01-01', '2019-10-01')]
    df_lst = []
    lst = []
    state_lst = []

    for symble in symble_lst:
        group = pd.read_csv('data/' + symble + '_' + period + '.csv').rename(columns={'date': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x))) \
            .assign(close_1=lambda df: df.close.shift(1))
        group_day = pd.read_csv('data/' + symble + '_' + '1440m' + '.csv')
        print(group)
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
                    .assign(ma_zd=lambda df: df['ma'] - df['ma'].shift(1)) \
                    .assign(atr=lambda df: df.atr.shift(1)) \
                    .merge(group, on=['date_time'], how='right') \
                    .sort_values(['date_time']).fillna(method='ffill')
                print(day_atr)
                for road_period_l in road_period_l_lst:
                    road_period_s = road_period_l
                    if len(day_atr) <= min(road_period_l, road_period_s):
                        continue
                    day_atr['h10'] = talib.MAX(day_atr['high'].shift(1).values, road_period_l)
                    day_atr['l10'] = talib.MIN(day_atr['low'].shift(1).values, road_period_s)
                    group__ = day_atr \
                        .assign(c_ma=lambda df: df.close_1 - df.ma).reset_index(drop=True)
                    # group__.to_csv('cl_trend/data/atr_' + symble + '.csv')
                    for (s_date, e_date) in date_lst:
                        group_ = group__[
                            (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)] \
                            .reset_index(drop=True)
                        if len(group_) < max(road_period_l, road_period_s, ma_period, N_ATR):
                            continue
                        back_stime = group_.at[0, 'date_time']
                        back_etime = group_.at[len(group_) - 1, 'date_time']

                        for ATR_n in ATR_n_lst:
                            for win_stop in win_stop_lst:
                                for status_day in status_days_lst:

                                    print(symble)
                                    signal_lst = []
                                    trad_times = 0
                                    net = 1
                                    net_lst = []
                                    pos_lst = []
                                    status = -10000  # 止损退出后的冷静天数
                                    pos = 0

                                    low_price_pre = 0
                                    high_price_pre = 100000000

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
                                                    hold_price.append(cost / (1 + fee))
                                                    high_price.append(cost / (1 + fee))
                                                    high_price.append(_row.high)
                                                    hold_price.append(_row.close)
                                                    net = (_row.close / cost) * net

                                                else:
                                                    net = net
                                            else:
                                                if (_row.high > _row.h10) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                                    cost = _row.h10 * (1 + fee)
                                                    if _row.open > _row.h10:
                                                        cost = _row.open * (1 + fee)
                                                    pos = 1
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost / (1 + fee))
                                                    high_price.append(cost / (1 + fee))
                                                    hold_price.append(_row.close)
                                                    high_price.append(_row.high)
                                                    net = _row.close / cost * net

                                                elif _row.high > high_price_pre:
                                                    s_time = _row.date_time
                                                    cost = high_price_pre * (1 + fee)
                                                    if max(_row.open, _row.low) > high_price_pre:
                                                        cost = max(_row.open, _row.low) * (1 + fee)
                                                    pos = 1
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost / (1 + fee))
                                                    high_price.append(cost / (1 + fee))
                                                    hold_price.append(_row.close)
                                                    high_price.append(_row.high)
                                                    net = (_row.close / cost) * net

                                                else:
                                                    net = net
                                        else:
                                            if pos == 1:
                                                # print(_row.date_time)
                                                # print(_row.low)
                                                # print(hold_price)
                                                # print(max(hold_price) - _row.atr * ATR_n)
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
                                                    pos = 0
                                                    high_price_pre = max(high_price)
                                                    net = net1 * net
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
                                                    status = 0
                                                elif _row.high >= cost * (1 + win_stop):

                                                    trad_times += 1
                                                    high_price.append(_row.high)
                                                    e_time = _row.date_time
                                                    s_price = cost * (1 + win_stop) * (1 - fee)
                                                    if min(_row.open, _row.high) >= cost * (1 + win_stop):
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

                                        net_lst.append(net)
                                        pos_lst.append(pos)
                                    # net_df = pd.DataFrame({'net': net_lst,
                                    #                        'date_time': group_.date_time.tolist(),
                                    #                        })
                                    # net_df.to_csv('cl_trend/data/data/spot_001_' + period + '_' +symble + '.csv')
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
                                        signal_lst, columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                             'max_ret', 'hold_day', 'position', 'bspk'])

                                    # signal_state.to_csv('cl_trend/data/signal_001_' + period + '_' + symble + '.csv')
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
                                    state_row.append(ma_period)
                                    state_row.append(win_stop)
                                    state_row.append(status_day)
                                    state_row.append(back_stime)
                                    state_row.append(back_etime)
                                    state_lst.append(state_row)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'road_period',
                                                    'ma_period', 'win_stop', 'status_day', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_tqa_eth_entry' + str(n) + '.csv')

