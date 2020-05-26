# coding=utf-8
'''
Created on 7.9, 2018
RSRS 策略 
@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
from dataapi import *
import pandas as pd
import talib as tb
import numpy as np
import statsmodels.api as sm


# style.use('ggplot')


# y=res_F['CAR']
# x=range(1,(len(res_F['CAR'])+1))
# X=sm.add_constant(x)
# result=(sm.OLS(y,X)).fit()
# k_F=result.params[1]

def cal_k(df_in, N):
    if N < N1:
        return [0, 0]
    else:
        x = df_in['low']
        y = df_in['high']
        X = sm.add_constant(x)
        result = (sm.OLS(y, X)).fit()
        k = result.params[1]
        r = result.rsquared
        return [k, r]


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    os.chdir(r'/Users/lion95/Documents/mywork/张芳的代码')
    t0 = time.time()
    # symble = 'yeeeth'
    x = 4
    symble_lst = 'btcusdt'
    n = 60  # 回测周期
    start = '2017-01-01 00:00'
    end = '2018-10-29 00:00'

    #    N1_lst=range(10,15) #  波动率周期
    #    N2_lst=range(15,20) # boll均线周期
    #    N3_lst=np.arange(0.001,0.005,0.001) # 波动率阈值
    #    N4_lst=[1,1.5] # boll轨道参数

    #    N1_lst=range(15,40) # 数据集大小
    #    N2_lst=range(0,10) # 一阶导数的阈值
    #    N3_lst=np.arange(0,0.6,0.1) #二阶导数的阈值

    N1_lst = [18]  # N1 计算线性回归的K值
    N2_lst = [600]  # N2 计算斜率标准化的周期
    N3_lst = [0.9]  # 开仓阈值

    #    N1_lst=range(15,30) # N1日最高最低价
    #    N2_lst=range(5,20)  # N2日最高价
    #    N3_lst=np.arange(0.01,0.06,0.01)

    #    N1_lst=[26] # N1日最高最低价
    #    N2_lst=[10]  # N2日最高价
    #    N3_lst=[0.05]

    N_ATR = 20
    ATR_n = 0.5
    fee = 0.004
    # s_date = '2016-01-01'
    # e_date = '2018-07-14'
    # group.loc[:, ['high', 'low', 'close', 'open']] = group.loc[:, ['high', 'low', 'close', 'open']]\
    #     .apply(lambda x: 1/x)
    # group = group.rename(columns={'high': 'low', 'low': 'high'})
    # print(group)
    df_lst = []
    lst = []
    state_lst = []

    # =============================================================================
    # 原来的数据提取
    # =============================================================================
    #    group = get_huobi_ontime_kline('btcusdt','60min', start, end)[2].reset_index() \
    #              .rename(columns={'date':'date_time'})
    #
    #
    #    group_day = get_huobi_ontime_kline(symble_lst,'1day',start,end)[2].reset_index()[
    #        ['date','high','low','close']]\
    #        .rename(columns={'date':'date_time'}) \
    #        .assign(date_time=lambda df: df.date_time.apply(lambda x: x[:10] + ' 00:00:00'))\
    #        .sort_values(['date_time'])

    group = pd.read_csv('data/btc_index_' + str(n) + 'm.csv').rename(columns={'date': 'date_time'}) \
        .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)))

    group = group[(group['date_time'] >= start) & (group['date_time'] <= end)] \
        .assign(day=lambda df: df.date_time.apply(lambda s: s[:10]))

    group_day = pd.read_csv('data/btc_index_' + '1440m' + '.csv').assign(
        date_time=lambda df: df.date_time + ' 00:00:00')
    group_day = group_day[(group_day['date_time'] >= start) & (group_day['date_time'] <= end)]

    group_day['atr'] = tb.ATR(group_day['high'].values, group_day['low'].values,
                              group_day['close'].values, N_ATR)
    day_atr = group_day[['date_time', 'atr']] \
        .assign(atr=lambda df: df.atr.shift(1)) \
        .merge(group, on=['date_time'], how='right') \
        .sort_values(['date_time']).fillna(method='ffill').reset_index(drop=True)

    #    for N_ATR in P_ATR_LST:
    #
    #        group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
    #                                     group_day['close'].values, N_ATR)
    #        day_atr = group_day[['date_time', 'atr']] \
    #            .assign(atr=lambda df: df.atr.shift(1)) \
    #            .merge(group, on=['date_time'], how='right') \
    #            .sort_values(['date_time']).fillna(method='ffill').drop(['index'], axis=1)
    #        for ATR_n in ATR_n_lst:
    for N1 in N1_lst:
        for N2 in N2_lst:
            for N3 in N3_lst:
                #                for N4 in N4_lst:
                #                    for count in count_lst:
                print(symble_lst)

                method = 'rsrs' + '_' + str(N1) + '_' + str(N2) + '_' + str(N3) + '_' + \
                         str(N_ATR) + '_' + str(ATR_n)
                print(method)
                signal_lst = []
                trad_times = 0
                if len(day_atr) > N1:
                    net = 1
                    net_lst = []
                    #                        group_ = day_atr\
                    #                            .assign(close_1=lambda df: df.close.shift(1)) \
                    #                            .assign(HH_s=lambda df: talib.MAX(df.high.values, N1))\
                    #                            .assign(LL_s=lambda df: talib.MIN(df.low.values, N1)) \
                    #                            .assign(HH_l=lambda df: talib.MAX(df.high.values, N2)) \
                    #                            .assign(LL_l=lambda df: talib.MIN(df.low.values, N2))\
                    #                            .assign(HH_s=lambda df: df.HH_s.shift(1)) \
                    #                            .assign(LL_s=lambda df: df.LL_s.shift(1)) \
                    #                            .assign(
                    #                            long_ratio=lambda df: ((df.close - df.LL_l) / (df.HH_l - df.LL_l) - 0.5) / 0.5)\
                    #                            .assign(long_ratio=lambda df: df.long_ratio.shift(1))

                    group_ = day_atr.copy()
                    #                group_['day']=group_.date_time.apply(lambda s:s[:10])
                    #                        group_=group_.assign(pivot1=lambda df:(df.open+df.low+df.close)/3)\
                    #                                     .assign(pivot2=lambda df:(df.open+df.low+2*df.close)/4)\
                    #                                     .assign(resis=lambda df:(2*df.pivot1-df.low))\
                    #                                     .assign(stat_sign=lambda df:[aapv(_row) for idx,_row in df.iterrows()])\
                    #                                     .assign(sign=lambda df:talib.MA(df['stat_sign'].values,N1))\
                    #                                     .assign(sign_1=lambda df:df.sign.shift(1))\
                    #                                     .assign(close_1=lambda df:df.close.shift(1))
                    group_.index = range(len(group_))

                    k_r_lst = [cal_k(group_.iloc[idx - N1:idx, :], idx) for idx, _row in group_.iterrows()]

                    group_ = group_.assign(KandR=k_r_lst) \
                        .assign(K=lambda df: df.KandR.apply(lambda s: s[0])) \
                        .assign(R=lambda df: df.KandR.apply(lambda s: s[1])) \
                        .assign(K_ma=lambda df: tb.MA(df['K'].values, N2)) \
                        .assign(K_delt=lambda df: tb.STDDEV(df['K'].values, timeperiod=N2, nbdev=1)) \
                        .assign(rsrs=lambda df: (df.K - df.K_ma) / df.K_delt) \
                        .assign(close_1=lambda df: df.close.shift(1)) \
                        .assign(rsrs_1=lambda df: df.rsrs.shift(1)) \
                        .assign(c_ma=lambda df: tb.MA(df['close'].values, 20)) \
                        .assign(c_ma_1=lambda df: df['c_ma'].shift(1)) \
                        .assign(c_ma_2=lambda df: df['c_ma'].shift(2)) \
                        .assign(rsrs_x=lambda df: df['rsrs'] * df['R']) \
                        .assign(rsrs_x_k=lambda df: df['rsrs_x'] * df['K']) \
                        .assign(rsrs_x_k_1=lambda df: df.rsrs_x_k.shift(1))

                    group_ = group_.dropna()
                    group_.index = range(len(group_))
                    # group_.to_csv('cl_trend/day_atr1.csv')
                    position = 0
                    high_price_pre = 1000000000
                    # signal_row = []
                    # stock_row = []
                    for idx, _row in group_.iterrows():

                        if (position == 0) & (_row.rsrs_x_k_1 > N3) & (_row.c_ma_1 > _row.c_ma_2) & (
                                _row.close_1 > _row.c_ma_1):
                            position = 1
                            s_time = _row.date_time
                            cost = _row.open
                            hold_price = []
                            high_price = []
                            hold_price.append(cost)
                            high_price.append(cost)
                            net = net * _row.close / cost
                        elif (position == 0) & (_row.high > high_price_pre):
                            position = 1
                            s_time = _row.date_time
                            cost = high_price_pre
                            hold_price = []
                            high_price = []
                            hold_price.append(cost)
                            high_price.append(cost)
                            net = net * _row.close / cost

                        elif position == 1:
                            if _row.low < max(hold_price) - _row.atr * ATR_n:
                                #                            if _row.rsrs_1< -N3:
                                position = 0
                                trad_times += 1
                                high_price.append(_row.high)
                                e_time = _row.date_time
                                s_price = max(hold_price) - _row.atr * ATR_n
                                if _row.open < max(hold_price) - _row.atr * ATR_n:
                                    s_price = _row.open
                                #                                s_price = _row.open
                                high_price_pre = max(high_price)
                                ret = s_price / cost - 1
                                signal_row = []
                                signal_row.append(s_time)
                                signal_row.append(e_time)
                                signal_row.append(cost)
                                signal_row.append(s_price)
                                signal_row.append(ret - fee)
                                signal_row.append(max(high_price) / cost - 1)
                                signal_row.append(len(hold_price))
                                net = net * s_price / _row.close_1 * (1 - fee)
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
                    plt.plot(net_lst)
                    signal_state = pd.DataFrame(signal_lst,
                                                columns=['s_time', 'e_time', 'b_price', 's_price', 'ret',
                                                         'max_ret', 'hold_day']) \
                        .assign(method=method)
                    signal_state.to_csv('result/signal_rsrs_' + str(n) + '_' + method + '.csv')
                    # df_lst.append(signal_state)
                    win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                    win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                    state_row = []
                    state_row.append(symble_lst)
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
                    state_row.append(mid_r)
                    state_row.append(win_R_3)
                    state_row.append(win_R_5)
                    state_row.append(ave_max)

                    state_row.append(N_ATR)
                    state_row.append(ATR_n)
                    state_row.append(N1)
                    state_row.append(N2)
                    state_row.append(N3)
                    #                    state_row.append(N4)

                    #                        state_row.append(K2)
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
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days', 'mid_r',
                                                    'win_r_3', 'win_r_5', 'ave_max', 'art_N', 'art_n', 'n1',
                                                    'n2', 'n3'])
    print(signal_state)
    signal_state.to_csv('result/state_rsrs' + '_' + str(n) + '_' + str(x) + '.csv')
#    signal_df = pd.concat(df_lst)
#    signal_df.to_csv('cl_trend/data/signal_tqa_' + str(n) + '_' + str(x) + '_.csv')
