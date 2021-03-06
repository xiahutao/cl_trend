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
style.use('ggplot')


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/zhangfang/PycharmProjects')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 8
    symble_lst = ['xbtusd', 'btcusdt', 'btcindex', 'ethusd', 'eosbtc']
    n = 60  # 回测周期
    period = '60m'

    N1_lst = [5, 6, 7]  # 短周期周期1
    N2_lst = [15, 25, 2]  # 长周期周期2
    
    K1_lst = [i/100 for i in range(65, 76, 5)]  # 近期比
    K2_lst = [i/100 for i in range(85, 96, 5)]  # 长期比例

    P_ATR_LST = [20]  # ATR周期
    ATR_n_lst = [0.5]  # ATR倍数
    win_stop_lst = [10]
    status_days_lst = [0]

    lever = 1
    fee = 0.003
    date_lst = [('2017-01-01', '2018-01-01'), ('2018-01-01', '2018-06-01'), ('2018-06-01', '2018-12-01')]
    # date_lst = [('2017-01-01', '2018-10-01')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        if symble == 'btcindex':
            group = pd.read_csv('data/btc_index_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x))) \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/btc_index_' + '1440m' + '.csv')
        elif symble == 'xbtusd':
            group = pd.read_csv('data/xbtusd_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x))) \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/xbtusd_' + '1440m' + '.csv')
        elif symble == 'ethusd':
            group = pd.read_csv('data/ethusdt_' + str(n) + 'm.csv') \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/ethusdt_1440m.csv')
        elif symble == 'eosbtc':
            group = pd.read_csv('data/eosbtc_' + str(n) + 'm.csv') \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/eosbtc_1440m.csv')
        else:
            group = pd.read_csv('data/btcusdt_' + period + '.csv').rename(columns={'date': 'date_time'}) \
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x))) \
                .assign(close_1=lambda df: df.close.shift(1))
            group_day = pd.read_csv('data/btcusdt_' + '1440m' + '.csv')
        for N_ATR in P_ATR_LST:
            if len(group_day) < N_ATR:
                continue
            group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                         group_day['close'].values, N_ATR)
            day_atr = group_day[['date_time', 'atr']]\
                .assign(atr=lambda df: df.atr.shift(1)) \
                .merge(group, on=['date_time'], how='right') \
                .sort_values(['date_time']).fillna(method='ffill')
            for N1 in N1_lst:
                for N2 in N2_lst:

                    if len(day_atr) <= min(N1, N2):
                        continue
                    if (N1 < N2):
                        group__ = day_atr \
                            .assign(HH_s1=lambda df: talib.MAX(df.high.values, N1)) \
                            .assign(LL_s1=lambda df: talib.MIN(df.low.values, N1)) \
                            .assign(HH_l1=lambda df: talib.MAX(df.high.values, N2)) \
                            .assign(LL_l1=lambda df: talib.MIN(df.low.values, N2)) \
                            .assign(HH_s1=lambda df: df.HH_s1.shift(1)) \
                            .assign(LL_s1=lambda df: df.LL_s1.shift(1)) \
                            .assign(long_ratio1=lambda df: ((df.close - df.LL_l1) / (
                                df.HH_l1 - df.LL_l1) - 0.5) / 0.5) \
                            .assign(long_ratio1=lambda df: df.long_ratio1.shift(1))
                        for (s_date, e_date) in date_lst:
                            group_ = group__[
                                (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)] \
                                .reset_index(drop=True)
                            print(group_)
                            back_stime = group_.at[0, 'date_time']
                            back_etime = group_.at[len(group_) - 1, 'date_time']

                            for K1 in K1_lst:
                                for K2 in K2_lst:

                                                for ATR_n in ATR_n_lst:
                                                    for win_stop in win_stop_lst:
                                                        for status_day in status_days_lst:
                                                            print(symble)
                                                            if (symble == 'ethusd') | (symble == 'eosbtc'):
                                                                status = -10000
                                                                signal_lst = []
                                                                trad_times = 0
                                                                net = 1
                                                                net_lst = []
                                                                pos_lst = []
                                                                pos = 0
                                                                low_price_pre = 0
                                                                high_price_pre = 100000000

                                                                for idx, _row in group_.iterrows():
                                                                    if pos == 0:
                                                                        if (status >= 0) & (
                                                                                status < status_day):
                                                                            status += 1
                                                                            if _row.high > high_price_pre:
                                                                                s_time = _row.date_time
                                                                                cost = high_price_pre * (
                                                                                            1 + fee)
                                                                                if max(_row.open,
                                                                                       _row.low) > high_price_pre:
                                                                                    cost = max(_row.open,
                                                                                               _row.low) * (
                                                                                                       1 + fee)
                                                                                pos = 1
                                                                                hold_price = []
                                                                                high_price = []
                                                                                hold_price.append(cost)
                                                                                high_price.append(cost)
                                                                                net = (_row.close / cost) * net

                                                                            else:
                                                                                net = net
                                                                        else:

                                                                            if (_row.long_ratio1 < -K2) & (
                                                                                        (_row.high - _row.LL_s1) > (_row.HH_s1 - _row.LL_s1) * (
                                                                                        0.5 + 0.5 * K1)):
                                                                                cost = ((0.5 + 0.5 * K1) * (_row.HH_s1 - _row.LL_s1) + _row.LL_s1) * (1 + fee)
                                                                                if _row.open > (0.5 + 0.5 * K1) * (
                                                                                        _row.HH_s1 - _row.LL_s1) + _row.LL_s1:
                                                                                    cost = _row.open * (1 + fee)
                                                                                pos = 1

                                                                                s_time = _row.date_time
                                                                                hold_price = []
                                                                                high_price = []
                                                                                hold_price.append(cost)
                                                                                high_price.append(cost)
                                                                                net = _row.close / cost * net

                                                                            elif (_row.high > high_price_pre) & (high_price_pre * 1.01 > _row.low):
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

                                                                            else:
                                                                                net = net
                                                                    else:
                                                                        if pos == 1:
                                                                            if _row.low < max(hold_price) - _row.atr * ATR_n:

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
                                                                                status = 0
                                                                                net = net1 * net
                                                                            elif _row.high >= cost * (
                                                                                    1 + win_stop):
                                                                                trad_times += 1
                                                                                high_price.append(_row.high)
                                                                                e_time = _row.date_time
                                                                                s_price = cost * (
                                                                                            1 + win_stop) * (
                                                                                                      1 - fee)
                                                                                if min(_row.open,
                                                                                       _row.high) >= cost * (
                                                                                        1 + win_stop):
                                                                                    s_price = min(_row.open,
                                                                                                  _row.high) * (
                                                                                                      1 - fee)
                                                                                net1 = s_price / _row.close_1
                                                                                ret = s_price / cost - 1
                                                                                signal_row = []
                                                                                signal_row.append(s_time)
                                                                                signal_row.append(e_time)
                                                                                signal_row.append(cost)
                                                                                signal_row.append(s_price)
                                                                                signal_row.append(ret)
                                                                                signal_row.append(
                                                                                    (max(
                                                                                        high_price) / cost) - 1)
                                                                                signal_row.append(
                                                                                    len(hold_price))
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
                                                                state_row.append(N1)
                                                                state_row.append(N2)
                                                                state_row.append(K1)
                                                                state_row.append(K2)
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
                                                                pos_lst = []
                                                                pos = 1
                                                                low_price_pre = 0
                                                                high_price_pre = 100000000
                                                                status = -10000
                                                                for idx, _row in group_.iterrows():
                                                                    if pos == 1:
                                                                        if (status >= 0) & (
                                                                                status < status_day):
                                                                            status += 1
                                                                            if _row.high > high_price_pre:
                                                                                s_time = _row.date_time
                                                                                cost = high_price_pre * (
                                                                                            1 + fee)
                                                                                if max(_row.open,
                                                                                       _row.low) > high_price_pre:
                                                                                    cost = max(_row.open,
                                                                                               _row.low) * (
                                                                                                       1 + fee)
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

                                                                            else:
                                                                                net = net
                                                                        else:

                                                                            if (_row.long_ratio1 < -K2) & (
                                                                                        (_row.high - _row.LL_s1) > (_row.HH_s1 - _row.LL_s1) * (
                                                                                        0.5 + 0.5 * K1)):
                                                                                cost = ((0.5 + 0.5 * K1) * (_row.HH_s1 - _row.LL_s1) + _row.LL_s1) * (1 + fee)
                                                                                if _row.open > (0.5 + 0.5 * K1) * (
                                                                                        _row.HH_s1 - _row.LL_s1) + _row.LL_s1:
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

                                                                            elif (_row.high > high_price_pre) & (high_price_pre * 1.01 > _row.low):
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
                                                                            else:
                                                                                net = net
                                                                    else:
                                                                        if pos > 1:
                                                                            if _row.low < max(hold_price) - _row.atr * ATR_n:

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
                                                                                status = 0
                                                                            elif _row.high >= cost * (
                                                                                    1 + win_stop):
                                                                                trad_times += 1
                                                                                high_price.append(_row.high)
                                                                                e_time = _row.date_time
                                                                                s_price = cost * (
                                                                                            1 + win_stop) * (
                                                                                                  1 - fee)
                                                                                if min(_row.open,
                                                                                       _row.high) >= cost * (
                                                                                        1 + win_stop):
                                                                                    s_price = min(_row.open,
                                                                                                  _row.high) * (
                                                                                                      1 - fee)
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
                                                                                    (pos * max(high_price) + (
                                                                                            1 - pos) * cost) / max(
                                                                                        high_price) - 1)
                                                                                signal_row.append(
                                                                                    len(hold_price))
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

                                                                    net_lst.append(net)
                                                                    pos_lst.append(pos)
                                                                # net_df = pd.DataFrame({'net': net_lst, 'date_time': group_.date_time.tolist(),
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
                                                                state_row.append(N1)
                                                                state_row.append(N2)
                                                                state_row.append(K1)
                                                                state_row.append(K2)

                                                                state_row.append(win_stop)
                                                                state_row.append(status_day)

                                                                state_row.append(back_stime)
                                                                state_row.append(back_etime)
                                                                state_lst.append(state_row)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'period_s1',
                                                    'period_l1', 'k_s1', 'k_l1', 'win_stop', 'status_day', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_ocm_reverse_btc_entry_' + str(n) + 'l.csv')

