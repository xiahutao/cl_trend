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
    n = 1440  # 回测周期
    period = '1440m'
    symble_lst = ['xbtusd', 'btcusdt', 'btcindex', 'ethusd', 'eosbtc']
    period_k0_lst = [i for i in range(15, 25, 1)]  # N1 计算线性回归的K值
    period_std0_lst = [i for i in range(40, 60, 3)]  # N2 计算斜率标准化的周期
    N_s_lst = [-0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1]  # 多头开仓阈值

    win_stop_lst = [10]
    status_days_lst = [0]

    P_ATR_LST = [20]  # ATR周期
    ATR_n_lst = [0.5]  # ATR倍数

    lever = 1
    fee = 0.003
    date_lst = [('2017-01-01', '2018-01-01'), ('2018-01-01', '2018-06-01'), ('2018-06-01', '2018-12-01')]
    # date_lst = [('2017-01-01', '2019-01-02')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        if symble == 'btcindex':
            group = pd.read_csv('data/btc_index_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))\
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/btc_index_' + '1440m' + '.csv')
        elif symble == 'xbtusd':
            group = pd.read_csv('data/xbtusd_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))\
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/xbtusd_' + '1440m' + '.csv')
        else:
            group = pd.read_csv('data/btcusdt_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))\
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/btcusdt_' + '1440m' + '.csv')
        for N_ATR in P_ATR_LST:
            if len(group_day) < N_ATR:
                continue
            group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                         group_day['close'].values, N_ATR)
            group_day['ma'] = talib.MA(group_day['close'].shift(1).values, 20)
            day_atr = group_day[['date_time', 'atr', 'ma']] \
                .assign(ma_zd=lambda df: df['ma'] - df['ma'].shift(1))\
                .assign(atr=lambda df: df.atr.shift(1)) \
                .merge(group, on=['date_time'], how='right') \
                .sort_values(['date_time']).fillna(method='ffill')\
                .reset_index(drop=True)
            print(day_atr)
            for period_k0 in period_k0_lst:
                if len(day_atr) <= period_k0:
                    continue
                k_lst0, r_lst0 = cal_k(day_atr.low.tolist(), day_atr.high.tolist(), period_k0)
                day_atr = day_atr.assign(K0=k_lst0)\
                    .assign(R0=r_lst0)
                for period_std0 in period_std0_lst:
                    group__ = day_atr.assign(K_ma0=lambda df: tb.MA(df['K0'].shift(1).values, period_std0)) \
                        .assign(rsrs0=lambda df: (df.K0.shift(1) - df.K_ma0) * df['R0'].shift(1) * df['K0'].shift(1) /
                                                 tb.STDDEV(df.K0.shift(1).values, timeperiod=period_std0, nbdev=1))\
                                .dropna().reset_index(drop=True)

                    for (s_date, e_date) in date_lst:
                        group_ = group__[
                            (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)] \
                            .reset_index(drop=True)
                        back_stime = group_.at[0, 'date_time']
                        back_etime = group_.at[len(group_) - 1, 'date_time']

                        for ATR_n in ATR_n_lst:

                            for N_s in N_s_lst:
                                for win_stop in win_stop_lst:
                                    for status_day in status_days_lst:
                                        if (symble == 'ethusd') | (symble == 'eosbtc'):
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
                                                        if _row.low < low_price_pre:
                                                            cost = low_price_pre * (1 - fee)
                                                            if min(_row.open,
                                                                   _row.high) < low_price_pre:
                                                                cost = min(_row.open, _row.high) * (
                                                                            1 - fee)
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

                                                        if (_row.rsrs0 < N_s) & (_row.close_1 < _row.ma) & (_row.ma_zd < 0):

                                                            cost = _row.open * (1 - fee)

                                                            pos = -1
                                                            s_time = _row.date_time
                                                            hold_price = []
                                                            low_price = []
                                                            hold_price.append(cost)
                                                            low_price.append(cost)
                                                            net = (2 - _row.close / cost) * net
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
                                                    if pos == -1:
                                                        if _row.high > min(hold_price) + _row.atr * ATR_n:
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
                                                            status = 0
                                                            net = net * net1
                                                            low_price_pre = min(low_price)
                                                        elif _row.low <= cost * (1 - win_stop):
                                                            trad_times += 1
                                                            e_time = _row.date_time
                                                            b_price = cost * (1 - win_stop) * (1 + fee)
                                                            if _row.open < cost * (1 - win_stop):
                                                                b_price = _row.open * (1 + fee)
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
                                            # net_df.to_csv('data/rsrs_' + symble + '_' + period + '.csv')
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
                                            state_row.append(period_k0)
                                            state_row.append(period_std0)

                                            state_row.append(N_s)

                                            state_row.append(win_stop)
                                            state_row.append(status_day)
                                            state_row.append(back_stime)
                                            state_row.append(back_etime)
                                            state_lst.append(state_row)
                                        else:
                                            signal_lst = []
                                            trad_times = 0
                                            net = 1
                                            net_lst = []
                                            # group_.to_csv('cl_trend/day_atr' + symble + '.csv')
                                            pos_lst = []
                                            pos = 1
                                            low_price_pre = 0
                                            high_price_pre = 100000000
                                            status = -10000
                                            for idx, _row in group_.iterrows():
                                                if pos == 1:
                                                    if (status >= 0) & (status < status_day):
                                                        status += 1
                                                        if _row.low < low_price_pre:
                                                            cost = low_price_pre * (1 - fee)
                                                            if min(_row.open,
                                                                   _row.high) < low_price_pre:
                                                                cost = min(_row.open, _row.high) * (
                                                                            1 - fee)
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
                                                        if (_row.rsrs0 < N_s) & (_row.close_1 < _row.ma) & (_row.ma_zd < 0):

                                                            cost = _row.open * (1 - fee)

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
                                                        elif _row.low < low_price_pre:
                                                            cost = low_price_pre * (1 - fee)
                                                            if min(_row.open,
                                                                   _row.high) < low_price_pre:
                                                                cost = min(_row.open, _row.high) * (
                                                                            1 - fee)
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
                                                        if _row.high > min(
                                                                hold_price) + _row.atr * ATR_n:
                                                            trad_times += 1
                                                            e_time = _row.date_time
                                                            b_price = (min(
                                                                hold_price) + _row.atr * ATR_n) * (
                                                                              1 + fee)
                                                            if max(_row.open, _row.low) > min(
                                                                    hold_price) + _row.atr * ATR_n:
                                                                b_price = max(_row.open,
                                                                              _row.low) * (1 + fee)
                                                            net1 = (pos * b_price + (
                                                                    1 - pos) * _row.close_1) / b_price
                                                            ret = (pos * b_price + (
                                                                        1 - pos) * cost) / b_price - 1
                                                            signal_row = []
                                                            signal_row.append(s_time)
                                                            signal_row.append(e_time)
                                                            signal_row.append(cost)
                                                            signal_row.append(b_price)
                                                            signal_row.append(ret)
                                                            signal_row.append(
                                                                (pos * min(low_price) + (
                                                                            1 - pos) * cost) / min(
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
                                                            status = 0
                                                        elif _row.low <= cost * (1 - win_stop):
                                                            trad_times += 1
                                                            e_time = _row.date_time
                                                            b_price = cost * (1 - win_stop) * (
                                                                    1 + fee)
                                                            if _row.open < cost * (1 - win_stop):
                                                                b_price = _row.open * (1 + fee)
                                                            net1 = (pos * b_price + (
                                                                    1 - pos) * _row.close_1) / b_price
                                                            ret = 1 - b_price / cost
                                                            signal_row = []
                                                            signal_row.append(s_time)
                                                            signal_row.append(e_time)
                                                            signal_row.append(cost)
                                                            signal_row.append(b_price)
                                                            signal_row.append(ret)
                                                            signal_row.append(
                                                                (pos * min(low_price) + (
                                                                        1 - pos) * cost) / min(
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
                                                signal_lst,
                                                columns=['s_time', 'e_time', 'b_price', 's_price',
                                                         'ret', 'max_ret', 'hold_day', 'position',
                                                         'bspk'])

                                            # signal_state.to_csv('cl_trend/data/signal_gzw_' + str(n) + '_' + symble + '_' + back_stime + 'ls.csv')
                                            # df_lst.append(signal_state)
                                            win_r, odds, ave_r, mid_r = get_winR_odds(
                                                signal_state.ret.tolist())
                                            win_R_3, win_R_5, ave_max = get_winR_max(
                                                signal_state.max_ret.tolist())
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
                                            state_row.append(period_k0)
                                            state_row.append(period_std0)

                                            state_row.append(N_s)

                                            state_row.append(win_stop)
                                            state_row.append(status_day)

                                            state_row.append(back_stime)
                                            state_row.append(back_etime)
                                            state_lst.append(state_row)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days', 'win_r_3',
                                                    'win_r_5', 'ave_max', 'art_N', 'art_n',
                                                    'period_k1', 'period_std1', 'N_l', 'win_stop', 'status_day',
                                                    's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_rsrs_btc_re_entry_' + str(n) + 's.csv')

