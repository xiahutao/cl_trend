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
    x = 7
    symble_lst = ['btcusdt']
    n = 240  # 回测周期
    period = '240m'

    N1_lst = [10, 12, 15, 17, 20]  # 短周期周期1
    N2_lst = [80, 100, 120, 140, 160]  # 长周期周期2

    K1_lst = [s / 100 for s in range(5, 96, 5)]  # 近期比例
    K2_lst = [s / 100 for s in range(5, 96, 5)]  # 长期比例

    P_ATR_LST = [20]  # ATR周期
    ATR_n_lst = [0.5]  # ATR倍数

    lever_lst = [1]  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever

    fee = 0.001
    date_lst = [('2017-01-01', '2017-10-01'), ('2017-10-01', '2018-01-01'), ('2018-01-01', '2018-11-01')]

    # group.loc[:, ['high', 'low', 'close', 'open']] = group.loc[:, ['high', 'low', 'close', 'open']]\
    #     .apply(lambda x: 1/x)
    # group = group.rename(columns={'high': 'low', 'low': 'high'})
    # print(group)
    df_lst = []
    lst = []

    state_lst = []
    for (s_date, e_date) in date_lst:

        for symble in symble_lst:
            if symble == 'ethusdt':
                # group = get_exsymbol_kline('BITFINEX', symble, period, s_date, e_date)[2] \
                #                    .loc[:, ['date_time', 'high', 'low', 'close', 'open']]
                #
                # group_day = get_exsymbol_kline('BITFINEX', symble, '1d', s_date, e_date)[2] \
                #         .loc[:, ['date_time', 'high', 'low', 'close', 'open']]
                group = pd.read_csv('data/ethusdt_' + str(n) + 'm.csv')
                group = group[(group['date_time'] >= s_date) & (group['date_time'] <= e_date)]
                group_day = pd.read_csv('data/ethusdt_1440m.csv')\
                    .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))
                group_day = group_day[(group_day['date_time'] >= s_date) & (group_day['date_time'] <= e_date)]
            else:
                group = pd.read_csv('data/btc_index_' + period + '.csv').rename(columns={'date': 'date_time'})\
                    .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))
                group = group[(group['date_time'] >= s_date) & (group['date_time'] <= e_date)]
                group_day = pd.read_csv('data/btc_index_' + '1440m' + '.csv').assign(
                    date_time=lambda df: df.date_time + ' 00:00:00')
                group_day = group_day[(group_day['date_time'] >= s_date) & (group_day['date_time'] <= e_date)]
                # print(group)
                # print(group_day)
            for N_ATR in P_ATR_LST:

                group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                             group_day['close'].values, N_ATR)
                day_atr = group_day[['date_time', 'atr']] \
                    .assign(atr=lambda df: df.atr.shift(1)) \
                    .merge(group, on=['date_time'], how='right') \
                    .sort_values(['date_time']).fillna(method='ffill').reset_index(drop=True)
                print(day_atr)
                back_stime = day_atr.at[0, 'date_time']
                back_etime = day_atr.at[len(day_atr) - 1, 'date_time']
                for ATR_n in ATR_n_lst:
                    for N1 in N1_lst:
                        for N2 in N2_lst:
                            if N1 < N2:
                                for K1 in K1_lst:
                                    for K2 in K2_lst:

                                        for lever in lever_lst:
                                            print(symble)
                                            method = 'OCM' + str(N1) + '_' + str(N2) + '_' + str(K1) + '_' + str(K2) + '_' + \
                                                     str(N_ATR) + '_' + str(ATR_n) + '_' + str(lever)
                                            print(method)
                                            signal_lst = []
                                            trad_times = 0
                                            if len(day_atr) > N2:
                                                net = 1
                                                net_lst = []
                                                group_ = day_atr\
                                                    .assign(close_1=lambda df: df.close.shift(1)) \
                                                    .assign(HH_s=lambda df: talib.MAX(df.high.values, N1))\
                                                    .assign(LL_s=lambda df: talib.MIN(df.low.values, N1)) \
                                                    .assign(HH_l=lambda df: talib.MAX(df.high.values, N2)) \
                                                    .assign(LL_l=lambda df: talib.MIN(df.low.values, N2))\
                                                    .assign(HH_s=lambda df: df.HH_s.shift(1)) \
                                                    .assign(LL_s=lambda df: df.LL_s.shift(1)) \
                                                    .assign(
                                                    long_ratio=lambda df: ((df.close - df.LL_l) / (df.HH_l - df.LL_l) - 0.5) / 0.5)\
                                                    .assign(long_ratio=lambda df: df.long_ratio.shift(1))

                                                # group_.to_csv('cl_trend/day_atr1.csv')
                                                pos = 1

                                                low_price_pre = 0
                                                high_price_pre = 100000000

                                                # signal_row = []
                                                # stock_row = []
                                                for idx, _row in group_.iterrows():

                                                    if pos == 1:
                                                        if (_row.long_ratio < -K2) & (
                                                                (_row.high - _row.LL_s) > (_row.HH_s - _row.LL_s) * (
                                                                0.5 + 0.5 * K1)):
                                                            cost = ((0.5 + 0.5 * K1) * (_row.HH_s - _row.LL_s) + _row.LL_s) * (1 - fee)
                                                            if _row.open > (0.5 + 0.5 * K1) * (
                                                                    _row.HH_s - _row.LL_s) + _row.LL_s:
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
                                                        elif _row.low < low_price_pre:
                                                            cost = low_price_pre * (1 - fee)
                                                            if _row.open < low_price_pre:
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
                                                        else:
                                                            net = net

                                                    elif pos == 0:

                                                        if _row.high > min(hold_price) + _row.atr * ATR_n:
                                                            trad_times += 1
                                                            e_time = _row.date_time
                                                            b_price = (min(hold_price) + _row.atr * ATR_n) * (1 + fee)
                                                            if _row.open > min(hold_price) + _row.atr * ATR_n:
                                                                b_price = _row.open * (1 + fee)
                                                            net1 = (pos * b_price + (1 - pos) * _row.close_1) / b_price
                                                            ret = 1 - b_price / cost
                                                            signal_row = []
                                                            signal_row.append(s_time)
                                                            signal_row.append(e_time)
                                                            signal_row.append(cost)
                                                            signal_row.append(b_price)
                                                            signal_row.append(ret)
                                                            signal_row.append((pos * min(low_price) + (1 - pos) * cost)/min(low_price) - 1)
                                                            signal_row.append(len(hold_price))
                                                            signal_row.append(pos)
                                                            signal_row.append('bp')
                                                            signal_lst.append(signal_row)
                                                            pos = 1
                                                            net2 = (pos * _row.close + (1 - pos) * b_price) / _row.close
                                                            low_price.append(_row.low)

                                                            net = net * net1 * net2
                                                            low_price_pre = min(low_price)

                                                        else:
                                                            low_price.append(_row.low)
                                                            hold_price.append(_row.close)
                                                            net = net * (pos * _row.close + (
                                                                        1 - pos) * _row.close_1) / _row.close
                                                    net_lst.append(net)

                                                ann_ROR = annROR(net_lst, n)
                                                total_ret = net_lst[-1]
                                                max_retrace = maxRetrace(net_lst)
                                                sharp = yearsharpRatio(net_lst, n)

                                                signal_state = pd.DataFrame(signal_lst,
                                                                            columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                                                     'max_ret', 'hold_day', 'position', 'bspk']) \
                                                    .assign(method=method)
                                                # signal_state.to_csv('cl_trend/data/signal_ocm_' + str(n) + '_' + symble + '_' + method + '.csv')
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
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'period_s',
                                                    'period_l', 'k_s', 'k_l', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('cl_trend/data/state_ocmup_'  + str(n) + 's' + str(x) + '.csv')
    # signal_df = pd.concat(df_lst)
    # signal_df.to_csv('cl_trend/data/signal_tqa_' + str(n) + '_' + str(x) + '_.csv')
