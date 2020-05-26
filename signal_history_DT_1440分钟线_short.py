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
    symble_lst = ['btcusdt']
    n = 30  # 回测周期
    period = '30m'

    road_period_lst = [20]  # 通道周期
    K1_lst = [0.5]  # 浮动倍数
    K2_lst = [0.8]  # 浮动倍数
    N1_lst = [130]  # 均线周期

    P_ATR_LST = [14]  # ATR周期
    ATR_n_lst = [0.8]  # ATR倍数

    lever = 1  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever
    fee = 0.001
    date_lst = [('2017-01-01', '2018-01-01'), ('2018-01-01', '2018-12-01')]
    # date_lst = [('2017-01-01', '2018-10-01')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        if symble == 'ethusdt':
            group = pd.read_csv('data/ethusdt_' + str(n) + 'm.csv')
            group = group[(group['date_time'] >= s_date) & (group['date_time'] <= e_date)]
            group_day = pd.read_csv('data/ethusdt_1440m.csv')
            group_day = group_day[(group_day['date_time'] >= s_date) & (group_day['date_time'] <= e_date)]
        elif symble == 'eosbtc':
            group = pd.read_csv('data/eosbtc_' + str(n) + 'm.csv')
            group = group[(group['date_time'] >= s_date) & (group['date_time'] <= e_date)]
            group_day = pd.read_csv('data/eosbtc_1440m.csv')
            group_day = group_day[(group_day['date_time'] >= s_date) & (group_day['date_time'] <= e_date)]
        else:
            group = pd.read_csv('data/btcusdt_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))
            group_day = pd.read_csv('data/btcusdt_' + '1440m' + '.csv')

        for N_ATR in P_ATR_LST:
            if len(group_day) == 0:
                continue
            group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                         group_day['close'].values, N_ATR)
            day_atr = group_day[['date_time', 'atr']] \
                .assign(atr=lambda df: df.atr.shift(1)) \
                .merge(group, on=['date_time'], how='right') \
                .sort_values(['date_time']).fillna(method='ffill').reset_index(drop=True)
            print(day_atr)

            for ATR_n in ATR_n_lst:
                for road_period in road_period_lst:
                    for K1 in K1_lst:
                        for K2 in K2_lst:
                            for N1 in N1_lst:
                                if len(group):
                                    print(symble)
                                    method = 'DT' + str(road_period) + '_' + str(K1) + '_' + str(K2) + \
                                             str(N_ATR) + '_' + str(ATR_n) + '_' + str(lever)
                                    print(method)

                                    if len(day_atr) > road_period:

                                        group__ = day_atr.assign(HH=lambda df: tb.MAX(df.high.values, road_period)) \
                                            .assign(HC=lambda df: tb.MAX(df.close.values, road_period)) \
                                            .assign(LL=lambda df: tb.MIN(df.low.values, road_period)) \
                                            .assign(LC=lambda df: tb.MIN(df.close.values, road_period))\
                                            .assign(max_HCL=lambda df: [max(row.HH - row.LC, row.HC - row.LL) for idx, row in df.iterrows()]) \
                                            .assign(buyRange_1=lambda df: K1 * df.max_HCL.shift(1) + df.open)\
                                            .assign(sellRange_1=lambda df: df.open - K2 * df.max_HCL.shift(1))\
                                            .assign(close_1=lambda df: df.close.shift(1)) \
                                            .assign(ma=lambda df: tb.MA(df.close.shift(1).values, N1))\
                                            .assign(c_ma=lambda df: df.close_1 - df.ma) \
                                            .assign(ma_zd=lambda df: df.ma - df.ma.shift(1)).dropna()

                                        for (s_date, e_date) in date_lst:
                                            net = 1
                                            net_lst = []
                                            signal_lst = []
                                            trad_times = 0
                                            group_ = group__[
                                                (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)]\
                                                .reset_index(drop=True)
                                            back_stime = group_.at[0, 'date_time']
                                            back_etime = group_.at[len(group_) - 1, 'date_time']

                                            if (symble == 'ethusdt') | (symble == 'eosbtc'):
                                                pos_lst = []
                                                pos = 0

                                                low_price_pre = 0
                                                high_price_pre = 100000000

                                                for idx, _row in group_.iterrows():
                                                    if pos == 0:

                                                        if (_row.high > _row.buyRange_1) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                                            cost = _row.buyRange_1 * (1 + fee)
                                                            pos = 1

                                                            s_time = _row.date_time
                                                            hold_price = []
                                                            high_price = []
                                                            hold_price.append(cost)
                                                            high_price.append(cost)
                                                            net = _row.close / cost * net

                                                        elif (_row.low < _row.sellRange_1) & (_row.c_ma < 0) & (_row.ma_zd < 0):

                                                            cost = _row.sellRange_1 * (1 - fee)

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
                                                            if (_row.low < _row.sellRange_1) & (_row.c_ma < 0) & (_row.ma_zd < 0):

                                                                s_price = _row.sellRange_1 * (1 - fee)
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
                                                                cost = _row.sellRange_1 * (1 - fee)

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
                                                            if (_row.high > _row.buyRange_1) & (_row.c_ma > 0) & (_row.ma_zd > 0):
                                                                b_price = _row.buyRange_1 * (1 + fee)

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

                                                                cost = _row.buyRange_1 * (1 + fee)

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
                                                    signal_lst, columns=['s_time', 'e_time', 'k_price', 'p_price', 'ret',
                                                                         'max_ret', 'hold_day', 'position', 'bspk']) \
                                                    .assign(method=method)
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
                                                state_row.append(road_period)
                                                state_row.append(K1)

                                                state_row.append(K2)
                                                state_row.append(N1)

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

                                                        if (_row.low < _row.sellRange_1) & (_row.c_ma < 0) & (
                                                                _row.ma_zd < 0):

                                                            cost = _row.sellRange_1 * (1 - fee)

                                                            net1 = (pos * cost + (1 - pos) * _row.close_1) / cost
                                                            pos = 0
                                                            net2 = (pos * _row.close + (1 - pos) * cost) / _row.close
                                                            s_time = _row.date_time
                                                            hold_price = []
                                                            low_price = []
                                                            hold_price.append(cost)
                                                            low_price.append(cost)
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
                                                        if pos == 0:
                                                            if (_row.high > _row.buyRange_1) & (_row.c_ma > 0) & (
                                                                    _row.ma_zd > 0):
                                                                b_price = _row.buyRange_1 * (1 + fee)

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
                                                                pos = 1 + lever

                                                                cost = _row.buyRange_1 * (1 + fee)

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
                                                    signal_lst, columns=['s_time', 'e_time', 'k_price', 'p_price',
                                                                         'ret', 'max_ret', 'hold_day', 'position', 'bspk']) \
                                                    .assign(method=method)
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
                                                state_row.append(road_period)
                                                state_row.append(K1)

                                                state_row.append(K2)
                                                state_row.append(N1)


                                                state_row.append(back_stime)
                                                state_row.append(back_etime)
                                                state_lst.append(state_row)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'road_period',
                                                    'K1', 'K2', 'N1', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_dt_' + str(n) + 'short.csv')

