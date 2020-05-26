# coding=utf-8

from backtest import *
import numpy as np
import pandas as pd
import talib
import warnings as warn


def calc_signal(df, param):
    # BITMEX 指数数据的特别处理
    df['high'] = df[['open', 'high']].max(axis=1)
    df['low'] = df[['open', 'low']].min(axis=1)

    df['ub'], df['md'], df['lb'] = talib.BBANDS(df['close'],
                                                timeperiod=param['boll_p'],
                                                nbdevup=param['boll_n_l'],
                                                nbdevdn=param['boll_n_s'])
    df['delta'] = np.sqrt(talib.VAR(df['close'], timeperiod=param['var_p'], nbdev=1))

    df['delta_pct'] = df['delta'] / df['md']

    # 昨天的 pct 上穿阈值，且昨天收盘价在 ub 之上
    df['buy_signal'] = ((df['delta_pct'].shift(2) < param['var_th_l'])
                        & (df['delta_pct'].shift(1) > param['var_th_l'])
                        & (df['close'].shift(1) > df['ub'].shift(1)))

    # 昨天的 pct 下穿阈值，且昨天收盘价在 lb 之下
    df['sell_signal'] = ((df['delta_pct'].shift(2) > param['var_th_s'])
                         & (df['delta_pct'].shift(1) < param['var_th_s'])
                         & (df['close'].shift(1) < df['lb'].shift(1)))

    df['buy_price'] = df['open']
    df['sell_price'] = df['open']

    # 计算日级别ATR用于atr止损
    df_d = resample(df, '1D')
    df_d['ATR'] = talib.ATR(df_d['high'], df_d['low'], df_d['close'], P_ATR).shift(1)
    # 第一天的数据不完整
    df_d = df_d.drop(df_d.index[0])

    df = df.merge(df_d[['date_time', 'ATR']], how='outer', on='date_time')
    df['ATR'] = df['ATR'].fillna(method='ffill')
    df['atr_retrace'] = df['ATR'] * ATR_N
    df.dropna(subset=['ATR'], inplace=True)

    return df


def main_strategy(df, param, capital):
    # capital: [tradecoin, basecoin]
    # price_method: 计价方式, 'tradecoin' or 'basecoin'
    tradecoin_amt = []
    basecoin_amt = []
    signal = []

    # 当前持仓期 close 的最低价或最高价，用作atr止损
    hold_c_minmax = np.nan
    hold_c_minmax_ls = []
    # 前一个多头持仓期的最高价，前一个空头持仓期的最低价，用作止损后突破前高开仓
    hold_h = np.nan
    hold_h_ls = []
    hold_l = np.nan
    hold_l_ls = []

    for rid, row in df.iterrows():
        pos_tmp = (capital[0] * row.open) / (capital[0] * row.open + capital[1])

        now = row.date_time

        # ATR止损，回测中优先判断止损，防止高估收益
        # 如果有做多仓位
        if pos_tmp > pos_base and row.low < hold_c_minmax - row.atr_retrace and not row.buy_signal:
            pos_tg = pos_base
            s_price = (hold_c_minmax - row.atr_retrace) * (1 - slippage)
            capital, s_price = order_pct_to(pos_tg, capital, s_price, fee)
            signal.append([now, 'sstop', s_price, pos_tg])
        # 如果有做空仓位
        elif pos_tmp < pos_base and row.high > hold_c_minmax + row.atr_retrace and not row.sell_signal:
            pos_tg = pos_base
            b_price = (hold_c_minmax + row.atr_retrace) * (1 + slippage)
            capital, b_price = order_pct_to(pos_tg, capital, b_price, fee)
            signal.append([now, 'bstop', b_price, pos_tg])

        # 次优先判断买入卖出信号，以开盘价交易
        elif row.buy_signal and pos_tmp <= pos_base:
            pos_tg = pos_base + 1
            b_price = row.buy_price * (1 + slippage)
            capital, b_price = order_pct_to(pos_tg, capital, b_price, fee)
            if pos_tmp < pos_base:
                # 如果要从仓位0买到仓位200%，必须输出两条信息：先从仓位0买到100%，再买到200%，用于计算多头空头收益
                signal.append([now, 'b0', b_price, pos_base])
            signal.append([now, 'b', b_price, pos_tg])
            if b_price < row.low:
                warn('impossible buy price', row)
            hold_h = hold_c_minmax = b_price
        elif row.sell_signal and pos_tmp >= pos_base:
            pos_tg = pos_base - 1
            s_price = row.sell_price * (1 - slippage)
            capital, s_price = order_pct_to(pos_tg, capital, s_price, fee)
            if pos_tmp > pos_base:
                signal.append([now, 's0', s_price, pos_base])
            signal.append([now, 's', s_price, pos_tg])
            if s_price > row.high:
                warn('impossible sell price', row)
            hold_l = hold_c_minmax = s_price

        # 最后判断重新入场条件，以重新入场的突破价交易
        elif pos_tmp == pos_base and row.high > hold_h:
            pos_tg = pos_base + 1
            b_price = hold_h * (1 + slippage)
            capital, b_price = order_pct_to(pos_tg, capital, b_price, fee)
            signal.append([now, 'breopen', b_price, pos_tg])
        elif pos_tmp == pos_base and row.low < hold_l:
            pos_tg = pos_base - 1
            s_price = hold_l * (1 - slippage)
            capital, s_price = order_pct_to(pos_tg, capital, s_price, fee)
            signal.append([now, 'sreopen', s_price, pos_tg])

        # 记录前次交易的低点高点用于再次开仓
        # 低点高点的计算方式为：开仓时的开仓价，以后k线的最高价或最低价
        # remark:今天过完才知道hold_h, hold_l 和 hold_c_min, hold_c_max
        if pos_tmp > pos_base:
            hold_h = np.nanmax([hold_h, row.high])
        elif pos_tmp < pos_base:
            hold_l = np.nanmin([hold_l, row.low])

        # 记录持仓状态下的最高价格、最低价格用作止损
        # 最高价最低价的计算方式为：开仓时的开仓价，开仓k线的收盘价，以后k线的收盘价
        pos_tmp = (capital[0] * row.open) / (capital[0] * row.open + capital[1])

        if pos_tmp > pos_base:
            hold_c_minmax = np.nanmax([hold_c_minmax, row.close])
        elif pos_tmp < pos_base:
            hold_c_minmax = np.nanmin([hold_c_minmax, row.close])
        else:
            hold_c_minmax = np.nan

        hold_c_minmax_ls.append(hold_c_minmax)
        hold_h_ls.append(hold_h)
        hold_l_ls.append(hold_l)

        # 记录持币数量
        tradecoin_amt = np.append(tradecoin_amt, capital[0])
        basecoin_amt = np.append(basecoin_amt, capital[1])

    # 计算净值、仓位
    tradecoin_net = tradecoin_amt + basecoin_amt / df['close']
    basecoin_net = tradecoin_amt * df['close'] + basecoin_amt
    pos = tradecoin_amt / tradecoin_net

    end_capital = capital

    net_df = pd.DataFrame({'date_time': df['date_time'],
                           'close': df['close'],
                           'net': eval(price_method+'_net'),
                           'pos': pos,
                           'tradecoin': tradecoin_amt,
                           'basecoin': basecoin_amt,
                           'close_minmax': hold_c_minmax_ls,
                           'hold_high': hold_h_ls,
                           'hold_low': hold_l_ls})

    signal_df = pd.DataFrame(signal, columns=['date_time', 'signal', 'price', 'pos'])

    if output_folder:
        df = df.merge(signal_df, how='outer', on='date_time')
        df = df.merge(net_df, how='outer', on='date_time')
        signal_df.to_csv(output_folder + '_signal.csv')
        df.to_csv(output_folder + '_net_signal.csv')

    return net_df, signal_df, end_capital


def summary(df, net_df, signal_df, param):
    long_ret, short_ret = long_short_ret(net_df, pos_base)
    month_ret = month_profit(net_df, pos_base)

    hold = mean_hold_k(net_df['pos'], pos_base)
    pos = mean_position(net_df['pos'])
    trad_time = trade_times(signal_df, pos_base)
    win_r, profit_r, profit_aver = win_profit_ratio(signal_df, pos_base)

    drawdown = max_drawdown(net_df['net'])

    if output_folder:
        param_str = period + '_' + symbol.replace('/', '_').replace('.', '') + str(param).replace(":", '').replace("'", '')

        fig, ax = plt.subplots(2, figsize=(25, 15))
        net_df_tmp = net_df.copy()
        net_df_tmp['close'] = net_df_tmp['close'] / net_df['close'].iloc[0]
        net_df_tmp['net'] = net_df_tmp['net'] / net_df['net'].iloc[0]

        net_df_tmp = net_df_tmp.merge(df[['date_time', 'delta_pct']], how='left', on='date_time')
        net_df_tmp.plot(x='date_time', y=['close', 'net', 'delta_pct'], secondary_y='delta_pct', title=param_str, grid=True, ax=ax[0])

        signal_df['price'] = signal_df['price'] / net_df['close'].iloc[0]
        try:
            signal_df[signal_df['signal'].str.startswith('b')].plot(x='date_time', y='price', c='lawngreen', style='.', ax=ax[0], label='buy',marker=6,ms=12)
            signal_df[signal_df['signal'].str.startswith('s')].plot(x='date_time', y='price', c='Red', style='.', ax=ax[0], label='sell',marker=7,ms=12)
        except:
            pass
        ax[0].grid(True, linestyle="-.")
        ax[0].grid(which='minor', axis='both')
        # ax[0].legend()

        ax[0].set_xlabel('')
        month_ret['month'] = month_ret['date_time'].dt.strftime('%Y-%m')
        month_ret.plot(kind='bar', x='month', y=['long_return', 'short_return'], color=['r', 'b'], grid=True, ax=ax[1])

        ax[1].legend()
        plt.tight_layout()
        fpath = mkfpath(output_folder + '_figure', param_str + '.png')
        plt.savefig(fpath)

    # 转换成日净值
    net_df.set_index('date_time', inplace=True)
    net_df = net_df.resample('1D').asfreq()
    net_df.reset_index(inplace=True)

    # 计算汇总
    net = net_df['net']
    date_time = net_df['date_time']
    base = net_df['close']
    tot_ret = total_ret(net)
    ann_ret = annual_ret(date_time, net)
    sharpe = sharpe_ratio(net)
    alpha, beta = alpha_beta(date_time, base, net)
    alpha_r = alpha / drawdown

    result = (symbol, param, tot_ret, ann_ret, sharpe,
              drawdown, alpha, beta, hold, pos,  alpha_r, trad_time, win_r,
              profit_r, long_ret, short_ret, profit_aver, period,
              net_df['date_time'].iloc[0], net_df['date_time'].iloc[-1])
    cols = ['symbol', 'param', 'tot_ret', 'ann_ret', 'sharpe',
            'max_down', 'alpha', 'beta', 'hold_k', 'position',
            'alpha_r', 'trades', 'win_r', 'profit_r', 'long_ret','short_ret',
            'profit_aver', 'period', 'start_time', 'end_time']
    return result, cols


def do_backtest(df, param, start_day, end_day, ini_capital):
    print('backtesting %s %s %s...' % (param, start_day, end_day))
    df = df[(df['date_time'] >= start_day) & (df['date_time'] < end_day)]

    net_df, signal_df, end_capital = main_strategy(df, param, ini_capital)
    return net_df, signal_df, end_capital


if __name__ == '__main__':
    exchange = 'BITMEX'
    symbol = '.bxbt'
    output_folder = ''
    df_ini = read_data_api(exchange, symbol, '1m', "2017-01-01", "2018-10-25")

    period = '1D'
    param_grid = {'var_p': [12],
                  'boll_p': [20],
                  'var_th_l': [0.04],
                  'boll_n_l': [0.6],
                  'var_th_s': [0.04],
                  'boll_n_s': [0.2]}
    period = '4h'
    param_grid = {'var_p': [8],
                  'boll_p': [20],
                  'var_th_l': [0.025],
                  'boll_n_l': [0.3],
                  'var_th_s': [0.015],
                  'boll_n_s': [0.6]}
    # period = '1h'
    # param_grid = {'var_p': [10],
    #               'boll_p': [40],
    #               'var_th_l': [0.005],
    #               'boll_n_l': [2],
    #               'var_th_s': [0.006],
    #               'boll_n_s': [1.2]}
    # period = '30m'
    # param_grid = {'var_p': [10],
    #               'boll_p': [30],
    #               'var_th_l': [0.001],
    #               'boll_n_l': [1],
    #               'var_th_s': [0.004],
    #               'boll_n_s': [1.2]}
    # period = '15m'
    # param_grid = {'var_p': [15],
    #               'boll_p': [40],
    #               'var_th_l': [0.001],
    #               'boll_n_l': [0.08],
    #               'var_th_s': [0.002],
    #               'boll_n_s': [0.6]}

    df_ini = resample(df_ini, period)
    param_ls = list(ParameterGrid(param_grid))

    ini_capital = [1, 0]
    pos_base = 1
    price_method = 'tradecoin'

    fee = 0.001
    slippage = 0.00
    P_ATR = 20
    ATR_N = 0.5

    date_ls = [('2017-01-01', '2017-12-01'),
               ('2017-12-01', '2018-04-01'),
               ('2018-04-01', '2018-10-25'),
               ('2017-01-01', '2018-10-25')]

    # date_ls = [('2017-01-01', '2018-10-25')]
    stat_ls = []

    for param in param_ls:
        for start_day, end_day in date_ls:
            staty_day = pd.to_datetime(start_day)
            end_day = pd.to_datetime(end_day)
            df = calc_signal(df_ini, param)
            net_df, signal_df, end_capital = do_backtest(df, param, start_day, end_day, ini_capital)
            stat, cols = summary(df, net_df, signal_df, param)
            stat_ls.append(stat)

    stat_df = pd.DataFrame(stat_ls, columns=cols)
    fname = output_folder + '_stat.csv'
    stat_df.to_csv(fname, float_format='%.4f')
    print(stat_df)
    print(stat_df[stat_df['alpha_r'] == stat_df['alpha_r'].max()])
