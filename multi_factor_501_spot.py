# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import time
import talib as tb
import datetime
import copy
import requests
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth',100)

retry_count = 3

prefix_url = 'https://mi.goupupupup.com'
# if socket.gethostname() == 'MBP-C':
#     prefix_url = 'http://127.0.0.1:8000'
# else:
#     prefix_url = 'https://mi.goupupupup.com'
exchange_url = prefix_url + '/data/exchanges'
exsymbol_url = prefix_url + '/data/symbols?exname=%s'
exkline_url = prefix_url + '/data/klines?exname=%s&symbol=%s&period=%s&starttime=%s&endtime=%s'
exkline_multi_url = prefix_url + '/data/klines/multi?exname=%s&symbols=%s&periods=%s&starttime=%s&endtime=%s'
exposition_url = prefix_url + '/data/positions?exname=%s&symbol=%s&timestamp=%s'
exfunding_url = prefix_url + '/data/fundings?exname=%s&symbol=%s&timestamp=%s'


def timestamp2str(ts):
    tmp = time.localtime(ts)
    # print tmp
    return time.strftime("%Y-%m-%d %H:%M:%S", tmp)


def get_exsymbol_kline_multi(exchange, symbols,periods,startstr,endstr):
    """
        获取交易所多币种多周期K线
        Parameters
        ------
          exchange:string
                    交易所名称
          symbols:string
                      币对列表，多个币对用逗号分隔
          periods：string
                      周期列表，多个周期用逗号分隔，1min,5min,15min,30min,60min,4hour,1day
          startstr:string
                      开始日期 format：YYYY-MM-DD  或  YYYY-MM-DD hh:mm
          endstr:string
                      结束日期 format：YYYY-MM-DD  或  YYYY-MM-DD hh:mm

        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          DataFrame:
              属性:日期 ，开盘价， 最高价， 收盘价， 最低价， 成交量
    """
    try:
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d")
        start = int(tmp.strftime("%s"))
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d")
        end = int(tmp.strftime("%s"))
    except Exception as e:
        try:
            tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d %H:%M")
            start = int(tmp.strftime("%s"))
            # print(tmp)
            # print(start)
            tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d %H:%M")
            end = int(tmp.strftime("%s"))
        except Exception as e:
            print(e)
            raise TypeError('ktype input error.')

    url = exkline_multi_url % (exchange, symbols,periods,start,end)
    for _ in range(retry_count):
        try:
            r = requests.get(url,timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode,errmsg,None

            df = pd.DataFrame(l['data']['items'], columns=['symbol','tickid','open','high','low','close','volume','amount'])

            df['date'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            cols.insert(0, cols.pop(cols.index('date')))
            # use ix to reorder
            df = df.ix[:, cols]


            return errcode,errmsg,df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def max_s(x, y):
    value_list = [a if a > b else b for a, b in zip(x, y)]
    return pd.Series(value_list, name="max")


def min_s(x, y):
    value_list = [a if a < b else b for a, b in zip(x, y)]
    return pd.Series(value_list, name="min")


def delay(df, period=1):
    return df.shift(period)


def sma(df, window=10):
    return df.rolling(window).mean()


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.open = pn_data['open']
        self.high = pn_data['high']
        self.low = pn_data['low']
        self.close = pn_data['close']
        self.volume = pn_data['volume']

    def alpha018(self):
        return self.close / delay(self.close, 5)

    def alpha050(self):
        data_mid1 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid1 = ts_sum(data_mid1, 12)
        data_mid2 = [0 if x <= y else z for x, y, z in zip(
            (self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(),
                                                                                 (self.low - delay(self.low)).abs())))]
        data_mid2 = pd.Series(data_mid2, name="values")
        data_mid2 = ts_sum(data_mid2, 12)
        data_mid3 = (data_mid2 - data_mid1) / (data_mid1 + data_mid2)
        return data_mid3

    def alpha052(self):
        data_mid1 = self.high - delay((self.high + self.low + self.close) / 3)
        data_mid1[data_mid1 < 0] = 0
        data_mid2 = delay((self.high + self.low + self.close) / 3) - self.low
        data_mid2[data_mid2 < 0] = 0
        return ts_sum(data_mid1, 26) / ts_sum(data_mid2, 26) * 100

    def alpha055(self):
        data_mid1 = (self.close - delay(self.close) + (self.close - self.open) / 2 + delay(self.close) - delay(
            self.open)) * 16

        data_mid_z = (self.high - delay(self.close)).abs() + (self.low - delay(self.close)).abs() / 2 + (
                    delay(self.close) - delay(self.open)).abs() / 4
        data_mid_vz = (self.low - delay(self.close)).abs() + (self.high - delay(self.close)).abs() / 2 + (
                    delay(self.close) - delay(self.open)).abs() / 4
        data_mid_vv = (self.high - delay(self.low)).abs() + (delay(self.close) - delay(self.open)) / 4

        data_mid_v = [vz if x1 > y1 and x2 > y2 else vv for x1, y1, x2, y2, vz, vv in
                      zip((self.low - delay(self.close)).abs(), (self.high - delay(self.low)).abs(),
                          (self.low - delay(self.close)).abs(), (self.high - delay(self.close)).abs(), data_mid_vz,
                          data_mid_vv)]
        data_mid2 = [z if x1 > y1 and x2 > y2 else v for x1, y1, x2, y2, z, v in
                     zip((self.high - delay(self.close)).abs(), (self.low - delay(self.close)).abs(),
                         (self.high - delay(self.close)).abs(), (self.high - delay(self.low)).abs(), data_mid_z,
                         data_mid_v)]
        data_mid3 = max_s((self.high - delay(self.close)).abs(), (self.low - delay(self.close)).abs())
        data_all = data_mid1 / data_mid2 * data_mid3
        return ts_sum(data_all, 20)

    def alpha060(self):
        data_mid1 = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        return ts_sum(data_mid1, 20)

    def alpha069(self):
        dtm = [0 if x <= y else z for x, y, z in
               zip(self.open, delay(self.open), max_s((self.high - self.open), (self.open - delay(self.open))))]
        dbm = [0 if x >= y else z for x, y, z in
               zip(self.open, delay(self.open), max_s((self.open - self.low), (self.open - delay(self.open))))]
        dtm = pd.Series(dtm, name="dtm")
        dbm = pd.Series(dbm, name="dbm")
        data_mid_z = (ts_sum(dtm, 20) - ts_sum(dbm, 20)) / ts_sum(dtm, 20)
        data_mid_vz = (ts_sum(dtm, 20) - ts_sum(dbm, 20)) / ts_sum(dbm, 20)
        data_mid_v = [0 if x == y else z for x, y, z in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_vz)]
        data_mid = [z if x > y else v for x, y, z, v in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_z, data_mid_v)]
        return pd.Series(data_mid, name="values")

    def alpha071(self):
        return (self.close - sma(self.close, 24)) / sma(self.close, 24) * 100

    def alpha003(self):
        data_mid1 = min_s(self.low, delay(self.close, 1))
        data_mid2 = max_s(self.high, delay(self.close, 1))
        data_mid3 = [z if x > y else v for x, y, z, v in zip(self.close, delay(self.close, 1), data_mid1, data_mid2)]
        data_mid3 = np.array(data_mid3)
        data_mid4 = self.close - data_mid3
        data_mid5 = [0 if x == y else z for x, y, z in zip(self.close, delay(self.close, 1), data_mid4)]
        data_mid5 = np.array(data_mid5)
        df = pd.Series(data_mid5, name="value")
        return ts_sum(df, 6)

    def alpha014(self):
        return self.close - delay(self.close, 5)

    def alpha051(self):
        data_mid4 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid4 = pd.Series(data_mid4, name="values")
        data_mid4 = ts_sum(data_mid4, 12)
        data_mid5 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),
                                                           (max_s((self.high - delay(self.high)).abs(),
                                                                  (self.low - delay(self.low)).abs())))]
        data_mid5 = pd.Series(data_mid5, name="values")
        data_mid5 = ts_sum(data_mid5, 12)
        data_mid6 = data_mid4 / (data_mid4 + data_mid5)
        return data_mid6

    def alpha128(self):
        data_mid1 = (self.high + self.low + self.close) / 3 * self.volume
        data_mid1[(self.high + self.low + self.close) / 3 <= delay((self.high + self.low + self.close) / 3)] = 0
        data_mid2 = (self.high + self.low + self.close) / 3 * self.volume
        data_mid2[(self.high + self.low + self.close) / 3 >= delay((self.high + self.low + self.close) / 3)] = 0
        return 100 - (100 / (1 + ts_sum((data_mid1), 14) / ts_sum((data_mid2), 14)))

    def alpha167(self):
        data_mid = self.close - delay(self.close)
        data_mid[self.close <= delay(self.close)] = 0
        return ts_sum(data_mid, 12) / self.close

    def alpha175(self):
        return sma(max_s(max_s((self.high - self.low), (delay(self.close) - self.high).abs()),
                         (delay(self.close) - self.low).abs()), 6) / self.close


symbols = ['ethbtc', 'eosbtc', 'etcbtc', 'iotabtc', 'iostbtc', 'ltcbtc', 'neobtc', 'trxbtc', 'xrpbtc', 'xlmbtc',
           'adabtc', 'ontbtc', 'bnbbtc', 'bchabcbtc', 'bchsvbtc', "mdabtc", "stratbtc", "xmrbtc", "dashbtc",
           "xembtc", "zecbtc", "wavesbtc", "btgbtc", "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc"]

trade_lst = ['ethbtc', 'xrpbtc', 'ltcbtc', 'eosbtc', 'bchabcbtc', 'bchsvbtc', 'bnbbtc', 'trxbtc', 'adabtc',
                 'ontbtc', 'etcbtc']

cols = ["period", "symbol", "tickid", "open", "high", "low", "close", "volume", "amount"]

alpha_test = ["Alpha.alpha018", "Alpha.alpha050", "Alpha.alpha052", "Alpha.alpha055",
              "Alpha.alpha060", "Alpha.alpha069", "Alpha.alpha071"]


def get_second_from_period(period):
    if period in ['15min', '30min', '60min']:
        return int(period[:2]) * 60
    elif period == '4hour':
        return 240 * 60
    else:
        return 1440 * 60


def get_m_from_period(period):
    if period == '4hour':
        return 5
    elif period == '1day':
        return 6
    else:
        return None


def get_max_coin(df_all, min_m, first_select_lst):
    df_all_ = df_all[df_all['tickid'] < df_all.tickid.max()]
    result_df = pd.DataFrame(columns=['tickid', 'alpha'])
    for symbol, df_temp in df_all_.groupby(['symbol']):
        df_temp = df_temp.reset_index(drop=True)
        result_dict_temp = []
        for alpha in alpha_test:
            Alpha = Alphas(df_temp)
            result_dict_temp.append(pd.DataFrame({'alpha': [alpha[6:]] * len(df_temp), 'tickid': df_temp.tickid.tolist(),
                                         symbol: eval(alpha)()}))
        result_df = result_df.merge(pd.concat(result_dict_temp), on=['tickid', 'alpha'], how='outer')
    result_df = result_df[result_df['tickid'] >= 1553443200]\
        .sort_values(by=['alpha', 'tickid'])\
        .set_index(['alpha', 'tickid'])

    result_df = result_df.rank(axis=1, numeric_only=True, na_option="keep", ascending=True)
    signal_lst = []
    for tickid, group in result_df.groupby(['tickid']):
        sers = group.sum()
        sers = sers[sers > 0]
        max_symble = list(sers.sort_values(ascending=False)[:min_m].index.values)
        select_symble = [i for i in max_symble if i in trade_lst and ((
                str(datetime.datetime.fromtimestamp(tickid))[:10], i) in first_select_lst)]
        signal_row = []
        signal_row.append(tickid)
        signal_row.append(select_symble)
        signal_lst.append(signal_row)
    signal_df = pd.DataFrame(signal_lst, columns=['tickid', 'signal']).sort_values(['tickid'])
    return signal_df


def get_predata(hqdata, maxholdday):
    df_lst = []
    for name, group in hqdata.groupby('symbol'):
        if group.shape[0] < 10:
            print("Stock {stk} has only {cnt} data and skip it.".format(stk=name, cnt=group.shape[0]))
            continue
        group = group.set_index('symbol')
        _df_lst = [group, ]
        df_shift_1 = group.loc[:, ['open', ]].shift(-1)
        for p in range(1, maxholdday + 1, 1):
            df_shift = group.loc[:, ['close', 'low', 'high']].shift(-p).rename(
                columns={"close": "c_p", 'low': "l_p", "high": "h_p"})
            df_shift_ = group.loc[:, ['close', ]].shift(-p + 1).rename(columns={"close": "c_p_"})
            _df = pd.concat([df_shift_1, df_shift, df_shift_], axis=1) \
                .assign(pre=lambda df: (df.c_p / df.open) - 1) \
                .assign(pre_next=lambda df: (df.c_p / df.c_p_ - 1)) \
                .assign(low=lambda df: (df.l_p / df.open - 1)) \
                .assign(high=lambda df: (df.h_p / df.open - 1)) \
                .rename(
                columns={'pre': 'pre{p}'.format(p=p), 'pre_next': 'pre{p}next'.format(p=p), 'low': 'low{p}'.format(p=p),
                         'high': 'high{p}'.format(p=p)}) \
                .drop(['c_p', 'l_p', 'h_p', 'open', 'c_p_'], axis=1)
            _df_lst.append(_df)
        df = pd.concat(_df_lst, axis=1)
        df_lst.append(df)
        # print(time.time() - t0)
    dealdataall = pd.concat(df_lst)
    dealdataall = dealdataall.reset_index(drop=False)
    return dealdataall


def weight_Df(signal_df, pre_df, weight, loss_stop, win_stop, fee):

    signal_slect = copy.deepcopy(signal_df)
    pre_data = copy.deepcopy(pre_df)
    date_lst = copy.deepcopy(signal_slect.tickid.tolist())
    pre_data = pre_data.set_index(['tickid', 'symbol'])
    signal_slect_ = signal_slect.set_index(['tickid'])
    lst = []
    row_dict = {}
    row_dict['tickid'] = date_lst[0]
    row_dict['weight'] = 0
    row_dict['symbol'] = []
    row_dict['return'] = 0
    lst.append(row_dict)
    sig_state_lst = []
    for idx, row_ in signal_slect_.iterrows():
        today = idx
        code_lst = copy.deepcopy(row_.signal)
        date_left = [int(i) for i in date_lst if i > today]
        if (len(date_left) >= 1) & (len(code_lst) > 0):
            for code in code_lst:
                # print(code)
                if date_left[0] in pd.DataFrame(lst).set_index(['tickid']).index:
                    try:
                        if code == pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol']:
                            # print('true1:', pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'])
                            continue
                    except Exception as e:
                        print(e)
                    try:
                        if code in pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'].tolist():
                            # print('true2:', pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'].tolist())
                            continue
                    except Exception as e:
                        print(e)
                code_hq_today = pre_data.loc[today, code]
                if code_hq_today['low1'] <= -loss_stop:
                    today_return = (1 - loss_stop) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (1 - loss_stop) * (1 - fee) * (1 - fee)-1
                    continue
                elif code_hq_today['high1'] >= win_stop:
                    today_return = (1 + win_stop) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (1 + win_stop) * (1 - fee) * (1 - fee)-1
                    continue
                elif code not in signal_slect_.loc[date_left[0]]['signal']:
                    today_return = (code_hq_today['pre1'] + 1) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (code_hq_today['pre1'] + 1) * (1 - fee) * (1 - fee)-1
                    continue
                today_return = (code_hq_today['pre1'] + 1) * weight * (1-fee)
                row_dict = {}
                row_dict['tickid'] = date_left[0]
                row_dict['weight'] = weight
                row_dict['symbol'] = code
                row_dict['return'] = today_return
                lst.append(row_dict)
                if len(date_left) > 1:
                    m = 1
                    for tickid in date_left[1:]:

                        m = m + 1
                        m2 = m - 1
                        prenextname = 'pre' + str(m) + 'next'
                        prename = 'pre' + str(m2)
                        lowprename = 'low' + str(m)
                        highprename = 'high' + str(m)
                        prelst = []
                        prelst.append(0.0)
                        for n in range(1, m):
                            premaxname = 'pre' + str(n)
                            prelst.append(code_hq_today[premaxname])
                        if code_hq_today[highprename] >= win_stop:
                            today_return = (1 + win_stop) / (1+code_hq_today[prename]) * (1 - fee) * weight
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = (win_stop + 1) * (1 - fee) * (1 - fee)-1
                            break
                        elif (code_hq_today[lowprename] - max(prelst))/(1+max(prelst)) < -loss_stop:
                            today_return = (1+max(prelst)) * (1-loss_stop)/(1+code_hq_today[prename]) * weight * (1-fee)
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = ((max(prelst) + 1) * (1-loss_stop)) * (1 - fee) * (1 - fee)-1
                            break
                        elif code not in signal_slect_.loc[tickid]['signal']:
                            today_return = (code_hq_today[prenextname] + 1) * (1 - fee) * weight
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = (code_hq_today['pre' + str(m)] + 1) * (1 - fee) * (1 - fee)-1
                            break
                        else:
                            today_return = (code_hq_today[prenextname] + 1) * weight
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = (code_hq_today['pre' + str(m)] + 1) * (1 - fee) * (1 - fee)-1
                    sig_state_row = []
                    sig_state_row.append(today)
                    sig_state_row.append(code)
                    sig_state_row.append(ret)
                    sig_state_row.append(m)
                    sig_state_lst.append(sig_state_row)
    weight_df = pd.DataFrame(lst)\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    sig_state_df = pd.DataFrame(sig_state_lst, columns=['tickid', 'symbol', 'ret', 'holddays'])\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    return weight_df, sig_state_df


def net_Df(weight_df, signal_slect):
    weight_df = weight_df.assign(tickid=lambda df: df.tickid.apply(lambda x: int(x)))
    weight_df = weight_df.set_index(['tickid'])
    date_lst = [int(i) for i in copy.deepcopy(signal_slect.tickid.tolist())]
    net = 1
    lst = []
    for tickid in date_lst:
        if tickid in weight_df.index:
            change = weight_df.loc[tickid]['return'].sum() + (1-weight_df.loc[tickid]['weight'].sum())

            pos = len(weight_df.loc[tickid]['symbol'])
            if pos > 3:
                pos = 1
        else:
            change = 1.0
            pos = 0
        row_ = []
        row_.append(tickid)
        row_.append(change)
        net = net * change
        row_.append(net)
        row_.append(pos)
        lst.append(row_)
    net_df = pd.DataFrame(lst, columns=['tickid', 'change', 'net', 'pos'])\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    return net_df


def run_multi_factor_501():
    strate_id = '501'

    period_lst = ['4hour', '1day']
    ma_period = 20
    now = datetime.datetime.now()
    start = (now - datetime.timedelta(days=50)).strftime('%Y-%m-%d')
    end = (now + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    symbol_lst = ','.join(symbols)
    errcode, errmsg, df_all_day = get_exsymbol_kline_multi('BIAN', symbol_lst, "1day", start, end)

    df_check = df_all_day.groupby('symbol').last()
    df_check = df_check[df_check['tickid'] < df_check['tickid'].max()]
    if not df_check.empty:
        print("有币对未获取最新K线,errmsg=" + str(df_check.index.tolist()) + str(df_check.tickid.values))
        return

    if errcode != 0:
        print("查询K线返回失败,errmsg=" + errmsg)
        return
    df_all_day[["open", "close", "high", "low", "volume"]] = df_all_day[["open", "close", "high", "low", "volume"]].astype(
        "float")
    c_ma_lst = []
    for symbol, df_ in df_all_day[['symbol', 'close', 'tickid']].groupby(['symbol']):
        c_ma = df_.assign(c_ma=lambda df: df['close'] - tb.MA(df['close'].values, ma_period))\
            .assign(date_time=lambda df: df.tickid.apply(lambda x: str(datetime.datetime.fromtimestamp(x))[:10]))\
            .assign(symbol=symbol)[['date_time', 'symbol', 'c_ma']]
        c_ma_lst.append(c_ma)
    c_ma_df = pd.concat(c_ma_lst)
    c_ma_df = c_ma_df[c_ma_df['c_ma'] > 0].set_index(['date_time', 'symbol'])
    first_selct = c_ma_df.index.tolist()

    for period in period_lst:
        second_per_period = get_second_from_period(period)
        max_m = get_m_from_period(period)
        if period == '1day':
            df_all = copy.deepcopy(df_all_day)
        else:
            errcode, errmsg, df_all = get_exsymbol_kline_multi('BIAN', symbol_lst, period, start, end)

            df_check = df_all.groupby('symbol').last()
            df_check = df_check[df_check['tickid'] < df_check['tickid'].max()]
            if not df_check.empty:
                print("有币对未获取最新K线,errmsg=" + str(df_check.index.tolist()) + str(df_check.tickid.values))
                return
            if errcode != 0:
                print("查询K线返回失败,errmsg="+errmsg)
                return
            df_all[["open", "close", "high", "low", "volume"]] = df_all[
                ["open", "close", "high", "low", "volume"]].astype("float")
        if int(time.time()) - second_per_period > df_all.tickid.max():
            print("所有币对未获取最新K线")
            return
        signal_df = get_max_coin(df_all, max_m, first_selct)  # 得到当前因子值最大的币对
        print("此时 %s 级别 %s 因子入选的币对为：%s" % (period, strate_id, signal_df.signal.tolist()[-1]))
        print("上一周期 %s 级别 %s 因子入选的币对为：%s" % (period, strate_id, signal_df.signal.tolist()[-2]))
        pre_data_df = get_predata(df_all[df_all['tickid'] < df_all.tickid.max()], 101)
        weight_df, sig_state_df = weight_Df(signal_df, pre_data_df, 1 / max_m, 1, 1000000, 0.002)
        net_df = net_Df(weight_df, signal_df)
        print("此时 %s 级别 %s 因子净值为：%s" % (period, strate_id, net_df.net.tolist()[-1]))
        print("上一周期 %s 级别 %s 因子净值为：%s" % (period, strate_id, net_df.net.tolist()[-2]))

    print("=========================")


if __name__ == '__main__':
    run_multi_factor_501()
