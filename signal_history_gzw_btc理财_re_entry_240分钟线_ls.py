# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
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
    symble_lst = ['btcindex']
    # symble_lst = ['ethusdt']
    n = 60  # 回测周期
    period = '60m'

    N1_lst = [3]  # 计数阈值
    N2_lst = [22]  # 统计周期
    N3_lst = [11]  # 计数上限

    N4_lst = [4]  # 计数阈值
    N5_lst = [6]  # 统计周期
    N6_lst = [20]  # 计数上限
    win_stop_lst = [10]
    status_days_lst = [184]
    P_ATR_LST = [12]  # ATR周期
    ATR_n_lst = [1.0]  # ATR倍数

    lever_lst = [1]
    fee = 0.003
    # date_lst = [('2017-01-01', '2018-01-01'), ('2018-01-01', '2018-06-01'), ('2018-06-01', '2018-12-01')]
    date_lst = [('2017-01-01', '2019-01-02')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        print(symble)
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
                .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))\
                .assign(close_1=lambda df: df.close.shift(1))
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
            for N2 in N2_lst:
                for N5 in N5_lst:
                    if len(day_atr) <= max(N5, N2):
                        continue
                    day_atr = day_atr \
                        .assign(hhv=lambda df: talib.MAX(df.high.shift(2).values, N2)) \
                        .assign(llv=lambda df: talib.MIN(df.low.shift(2).values, N5)).dropna()
                    HHV_nums = 0
                    LLV_nums = 0

                    hhv_num_list = []
                    llv_num_list = []
                    for idx, irow in day_atr.iterrows():
                        if irow['close_1'] > irow['hhv']:
                            HHV_nums = HHV_nums + 1
                            LLV_nums = LLV_nums - 1
                        elif irow['close_1'] < irow['llv']:
                            HHV_nums = HHV_nums - 1
                            LLV_nums = LLV_nums + 1
                        hhv_num_list.append(HHV_nums)
                        llv_num_list.append(LLV_nums)
                    day_atr = day_atr.assign(hhv_num=hhv_num_list) \
                        .assign(llv_num=llv_num_list)
                    for N3 in N3_lst:
                        for N6 in N6_lst:
                            if len(day_atr) <= max(N3, N6):
                                continue
                            group__ = day_atr\
                                .assign(hhv_num_chg=lambda df: df.hhv_num - df.hhv_num.shift(N3)) \
                                .assign(llv_num_chg=lambda df: df.llv_num - df.llv_num.shift(N6))
                            for (s_date, e_date) in date_lst:
                                group_ = group__[
                                    (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)].dropna() \
                                    .reset_index(drop=True)
                                if len(group_) < 20:
                                    continue
                                back_stime = group_.at[0, 'date_time']
                                back_etime = group_.at[len(group_) - 1, 'date_time']
                                for ATR_n in ATR_n_lst:
                                    for N1 in N1_lst:
                                        for N4 in N4_lst:
                                            for win_stop in win_stop_lst:
                                                for status_day in status_days_lst:
                                                    print(symble)
                                                    if (symble == 'ethusd') | (symble == 'eosbtc'):
                                                        signal_lst = []
                                                        trad_times = 0
                                                        net = 1
                                                        net_lst = []
                                                        pos = 0
                                                        low_price_pre = 0
                                                        high_price_pre = 100000000
                                                        pos_lst = []
                                                        status = -10000
                                                        for idx, _row in group_.iterrows():
                                                            if pos == 0:
                                                                if (status >= 0) & (
                                                                        status < status_day):
                                                                    status += 1
                                                                    if _row.high > high_price_pre:
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
                                                                    if _row.hhv_num_chg > N1:
                                                                        cost = max(_row.open, _row.low) * (1 + fee)

                                                                        pos = 1

                                                                        s_time = _row.date_time
                                                                        hold_price = []
                                                                        high_price = []
                                                                        hold_price.append(cost)
                                                                        high_price.append(cost)
                                                                        net = _row.close / cost * net
                                                                    elif _row.llv_num_chg > N4:
                                                                        cost = min(_row.open, _row.high) * (1 - fee)

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
                                                                    if _row.llv_num_chg > N4:

                                                                        s_price = min(_row.open, _row.high) * (1 - fee)
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
                                                                        signal_row.append((max(high_price) / cost) - 1)
                                                                        signal_row.append(len(hold_price))
                                                                        signal_row.append(pos)
                                                                        signal_row.append('spk')
                                                                        signal_lst.append(signal_row)
                                                                        s_time = _row.date_time
                                                                        cost = min(_row.open, _row.high)

                                                                        pos = -1
                                                                        net2 = (cost - _row.close) / cost + 1
                                                                        hold_price = []
                                                                        low_price = []
                                                                        hold_price.append(cost)
                                                                        low_price.append(cost)
                                                                        net = net1 * net2 * net
                                                                    elif (_row.hhv_num_chg <= N1) & (_row.low < max(hold_price) - _row.atr * ATR_n):

                                                                        trad_times += 1
                                                                        high_price.append(_row.high)
                                                                        e_time = _row.date_time
                                                                        s_price = (max(hold_price) - _row.atr * ATR_n) * (
                                                                                    1 - fee)
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

                                                                elif pos == -1:
                                                                    if _row.hhv_num_chg > N1:
                                                                        b_price = max(_row.open, _row.low) * (1 + fee)
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
                                                                        signal_row.append((cost - min(low_price)) / cost)
                                                                        signal_row.append(len(hold_price))
                                                                        signal_row.append(pos)
                                                                        signal_row.append('bpk')
                                                                        signal_lst.append(signal_row)
                                                                        pos = 1

                                                                        cost = max(_row.open, _row.low) * (1 + fee)

                                                                        net2 = _row.close / cost
                                                                        s_time = _row.date_time
                                                                        hold_price = []
                                                                        high_price = []
                                                                        hold_price.append(cost)
                                                                        high_price.append(cost)
                                                                        net = net1 * net2 * net

                                                                    elif (_row.llv_num_chg <= N4) & (_row.high > min(hold_price) + _row.atr * ATR_n):
                                                                        trad_times += 1
                                                                        e_time = _row.date_time
                                                                        b_price = (min(hold_price) + _row.atr * ATR_n) * (1 + fee)
                                                                        if max(_row.open, _row.low) > min(hold_price) + _row.atr * ATR_n:
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
                                                                        status = 0
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

                                                        signal_state = pd.DataFrame(signal_lst,
                                                                                    columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
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
                                                        state_row.append(N3)
                                                        state_row.append(N4)
                                                        state_row.append(N5)
                                                        state_row.append(N6)
                                                        state_row.append(win_stop)
                                                        state_row.append(status_day)
                                                        state_row.append(back_stime)
                                                        state_row.append(back_etime)
                                                        state_lst.append(state_row)
                                                    else:
                                                        signal_lst = []
                                                        pos = 1
                                                        net = 1
                                                        net_lst = []
                                                        low_price_pre = 0
                                                        high_price_pre = 100000000
                                                        pos_lst = []
                                                        trad_times = 0
                                                        status = -10000
                                                        for idx, _row in group_.iterrows():
                                                            if pos == 1:
                                                                if (status >= 0) & (status < status_day):
                                                                    status += 1
                                                                    if _row.high > high_price_pre:
                                                                        s_time = _row.date_time
                                                                        cost = high_price_pre * (1 + fee)
                                                                        if max(_row.open, _row.low) > high_price_pre:
                                                                            cost = max(_row.open, _row.low) * (1 + fee)
                                                                        pos = 2
                                                                        hold_price = []
                                                                        high_price = []
                                                                        hold_price.append(cost)
                                                                        high_price.append(cost)
                                                                        net = (_row.close / cost) * net
                                                                    elif _row.low < low_price_pre:
                                                                        cost = low_price_pre * (1 - fee)
                                                                        if min(_row.open, _row.high) < low_price_pre:
                                                                            cost = min(_row.open, _row.high) * (1 - fee)
                                                                        pos = 0
                                                                        s_time = _row.date_time
                                                                        hold_price = []
                                                                        low_price = []
                                                                        hold_price.append(cost)
                                                                        low_price.append(cost)
                                                                        net = ((cost - _row.close) / cost + 1) * net
                                                                    else:
                                                                        net = net
                                                                else:
                                                                    if _row.hhv_num_chg > N1:
                                                                        cost = max(_row.open, _row.low) * (1 + fee)
                                                                        net1 = (pos * cost + (
                                                                                    1 - pos) * _row.close_1) / cost
                                                                        pos = 2
                                                                        net2 = (pos * _row.close + (
                                                                                    1 - pos) * cost) / _row.close
                                                                        s_time = _row.date_time
                                                                        hold_price = []
                                                                        high_price = []
                                                                        hold_price.append(cost)
                                                                        high_price.append(cost)
                                                                        net = net1 * net2 * net

                                                                    elif _row.llv_num_chg > N4:

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
                                                                    if _row.llv_num_chg > N4:
                                                                        s_price = min(_row.open, _row.high) * (1 - fee)
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
                                                                            (pos * max(high_price) + (
                                                                                        1 - pos) * cost) / max(
                                                                                high_price) - 1)
                                                                        signal_row.append(len(hold_price))
                                                                        signal_row.append(pos)
                                                                        signal_row.append('spk')
                                                                        signal_lst.append(signal_row)
                                                                        s_time = _row.date_time
                                                                        cost = min(_row.open, _row.high)
                                                                        pos = 0
                                                                        net2 = (pos * _row.close + (1 - pos) * cost) / _row.close
                                                                        hold_price = []
                                                                        low_price = []
                                                                        hold_price.append(cost)
                                                                        low_price.append(cost)
                                                                        net = net1 * net2 * net
                                                                    elif (_row.hhv_num_chg <= N1) & (_row.low < max(
                                                                            hold_price) - _row.atr * ATR_n):
                                                                        trad_times += 1
                                                                        high_price.append(_row.high)
                                                                        e_time = _row.date_time
                                                                        s_price = (max(hold_price) - _row.atr * ATR_n) * (1 - fee)
                                                                        if min(_row.open, _row.high) < max(
                                                                                hold_price) - _row.atr * ATR_n:
                                                                            s_price = min(_row.open, _row.high) * (
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
                                                                        signal_row.append((pos * max(high_price) + (
                                                                                    1 - pos) * cost) / max(
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
                                                                    elif _row.high >= cost * (1 + win_stop):
                                                                        trad_times += 1
                                                                        high_price.append(_row.high)
                                                                        e_time = _row.date_time
                                                                        s_price = cost * (1 + win_stop) * (
                                                                                1 - fee)
                                                                        if min(_row.open, _row.high) >= cost * (
                                                                                1 + win_stop):
                                                                            s_price = min(_row.open, _row.high) * (
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
                                                                            (pos * max(high_price) + (
                                                                                        1 - pos) * cost) / max(
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
                                                                        pos = pos * _row.close / (pos * _row.close + (
                                                                                    1 - pos) * _row.close_1)
                                                                elif pos == 0:
                                                                    if _row.hhv_num_chg > N1:
                                                                        b_price = max(_row.open, _row.low) * (1 + fee)
                                                                        e_time = _row.date_time
                                                                        trad_times += 1
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
                                                                        signal_row.append('bpk')
                                                                        signal_lst.append(signal_row)
                                                                        pos = 2

                                                                        cost = max(_row.open, _row.low) * (1 + fee)

                                                                        net2 = (pos * _row.close + (
                                                                                    1 - pos) * cost) / _row.close
                                                                        s_time = _row.date_time
                                                                        hold_price = []
                                                                        high_price = []
                                                                        hold_price.append(cost)
                                                                        high_price.append(cost)
                                                                        net = net1 * net2 * net

                                                                    elif (_row.llv_num_chg <= N4) & (_row.high > min(
                                                                            hold_price) + _row.atr * ATR_n):
                                                                        trad_times += 1
                                                                        e_time = _row.date_time
                                                                        b_price = (min(
                                                                            hold_price) + _row.atr * ATR_n) * (1 + fee)
                                                                        if max(_row.open, _row.low) > min(
                                                                                hold_price) + _row.atr * ATR_n:
                                                                            b_price = max(_row.open, _row.low) * (
                                                                                        1 + fee)
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
                                                                        signal_row.append((pos * min(low_price) + (
                                                                                    1 - pos) * cost) / min(low_price) - 1)
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
                                                                        net1 = (pos * b_price + (1 - pos) * _row.close_1) / b_price
                                                                        ret = 1 - b_price / cost
                                                                        signal_row = []
                                                                        signal_row.append(s_time)
                                                                        signal_row.append(e_time)
                                                                        signal_row.append(cost)
                                                                        signal_row.append(b_price)
                                                                        signal_row.append(ret)
                                                                        signal_row.append((pos * min(low_price) + (
                                                                                    1 - pos) * cost) / min(low_price) - 1)
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
                                                        net_df = pd.DataFrame({'net': net_lst,
                                                                               'date_time': group_.date_time.tolist(),
                                                                               'close': group_.close.tolist(),
                                                                               'pos': pos_lst}).assign(
                                                            close=lambda df: df.close / df.close.tolist()[0])
                                                        net_df.to_csv('data/btc_gzw_btcindex_' + period + '.csv')
                                                        # net_df[['date_time', 'net', 'close']].plot(
                                                        #     x='date_time', kind='line', grid=True,
                                                        #     title=symble + '_' + period)
                                                        #
                                                        # plt.show()
                                                        ann_ROR = annROR(net_lst, n)
                                                        total_ret = net_lst[-1]
                                                        max_retrace = maxRetrace(net_lst)
                                                        sharp = yearsharpRatio(net_lst, n)

                                                        signal_state = pd.DataFrame(signal_lst,
                                                                                    columns=['s_time', 'e_time',
                                                                                             'b_price', 's_price',
                                                                                             'ret',
                                                                                             'max_ret', 'hold_day',
                                                                                             'position', 'bspk'])
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
                                                        state_row.append(N1)
                                                        state_row.append(N2)

                                                        state_row.append(N3)
                                                        state_row.append(N4)
                                                        state_row.append(N5)

                                                        state_row.append(N6)
                                                        state_row.append(win_stop)
                                                        state_row.append(status_day)
                                                        state_row.append(back_stime)
                                                        state_row.append(back_etime)
                                                        state_lst.append(state_row)
                                                        # print('胜率=', win_r)
                                                        # print('盈亏比=', odds)
                                                        # print('总收益=', total_ret - 1)
                                                        # print('年化收益=', ann_ROR)
                                                        # print('夏普比率=', sharp)
                                                        # print('最大回撤=', max_retrace)
                                                        # print('交易次数=', len(signal_state))
                                                        # print('平均每次收益=', ave_r)
                                                        # print('平均持仓周期=', signal_state.hold_day.mean())
                                                        # print('中位数收益=', mid_r)
                                                        # print('超过3%胜率=', win_R_3)
                                                        # print('超过5%胜率=', win_R_5)
                                                        # print('平均最大收益=', ave_max)
                                                        # print('参数=', method)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'N1',
                                                    'N2', 'N3', 'N1s', 'N2s', 'N3s', 'win_stop', 'status_day', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_gzw_usdt_re_entry' + str(n) + 'ls.csv')
    # signal_df = pd.concat(df_lst)
    # signal_df.to_csv('cl_trend/data/signal_tqa_' + str(n) + '_' + str(x) + '_.csv')
