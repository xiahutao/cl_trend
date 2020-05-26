# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

import json
import datetime, time
import requests
import pandas as pd

# http://47.74.16.216:9000/exchanges?datatype=1
# http://47.74.16.216:9000/symbols?exname=BITFINEX&datatype=1
# http://47.74.16.216:9000/kline?exname=BIAN&symbol=btcusdt&period=1d&starttime=1500825600&endtime=1532361600


retry_count = 3

exchange_url = 'http://47.74.16.216:9000/exchanges?datatype=%d'
exsymbol_url = 'http://47.74.16.216:9000/symbols?exname=%s&datatype=%d'
exkline_url = 'http://47.74.16.216:9000/kline?exname=%s&symbol=%s&period=%s&starttime=%s&endtime=%s'
huobi_kline_url = 'https://mi.goupupupup.com/klines?symbol=%s&period=%s&starttime=%s&endtime=%s'
huobi_exchange_url = 'https://mi.goupupupup.com/klines/symbols'


def get_exchange(datatype=1):
    """
        获取交易所列表
        Parameters
        ------
          datatype:int
                    数据类型，取值  1-K线
        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          List:
              币对列表
    """

    for _ in range(retry_count):
        try:
            r = requests.get(exchange_url % datatype, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None
            return errcode, errmsg, l['data']['items']
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_exsymbol(exchange, datatype=1):
    """
        获取交易所币对
        Parameters
        ------
          exchange:string
                    交易所名称
          datatype:int
                    数据类型，取值  1-K线

        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          List:
              币对列表
    """
    url = exsymbol_url % (exchange, datatype)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None
            return errcode, errmsg, l['data']['items']
        except Exception as e:
            print(e)
    raise IOError("无法连接")


def timestamp2str(ts):
    tmp = time.localtime(ts)
    # print tmp
    return time.strftime("%Y-%m-%d %H:%M:%S", tmp)


def get_exsymbol_kline(exchange, symbol, period, startstr, endstr):
    """
        获取交易所币对K线
        Parameters
        ------
          exchange:string
                      交易所名称
          symbol:string
                      币对名称
          period：string
                      周期，1m,5m,15m,30m,1h,4h,1d    m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
          startstr:string
                      开始日期 format：YYYY-MM-DD
          endstr:string
                      结束日期 format：YYYY-MM-DD

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
        print(e)
        raise TypeError('ktype input error.')

    url = exkline_url % (exchange, symbol, period, start, end)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None

            df = pd.DataFrame(l['data']['items'], columns=['tickid', 'open', 'high', 'low', 'close', 'volume'])

            df['date_time'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            cols.insert(0, cols.pop(cols.index('date_time')))
            # use ix to reorder
            df = df.ix[:, cols]

            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_huobi_ontime_kline(symbol, period, startstr, endstr):
    """
        获取交易所币对K线
        Parameters
        ------
          symbol:string
                      币对名称
          period：string
                      周期，1min,5min,15min,30min,60min,4hour,1day    m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
          startstr:int
                      开始日期 format：YYYY-MM-DD HH:MM
          endstr:string
                      结束日期 format：YYYY-MM-DD HH:MM

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
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d %H:%M")
        start = int(tmp.strftime("%s"))
        print(tmp)
        print(start)
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d %H:%M")
        end = int(tmp.strftime("%s"))
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    url = huobi_kline_url % (symbol, period, start, end)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None

            df = pd.DataFrame(l['data']['items'], columns=['tickid', 'open', 'high', 'low', 'close', 'volume'])

            df['date_time'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            cols.insert(0, cols.pop(cols.index('date_time')))
            # use ix to reorder
            df = df.ix[:, cols]

            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_huobi_exchange():
    """
        获取交易所币对

    """

    url = huobi_exchange_url
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None

            df = pd.DataFrame(l['data']['items'], columns=['symbolname'])

            return errcode, errmsg, df.symbolname.tolist()
        except Exception as e:
            print(e)

    raise IOError("无法连接")


if __name__ == '__main__':
    errcode, errmsg, result = get_exchange()
    print(result)

    errcode, errmsg, result = get_exsymbol("BITFINEX")
    print(result)
    pd.DataFrame(list(result)).to_csv('bitfinex.csv')
    df = get_exsymbol_kline("BITFINEX", "btcusdt", "30m", "2017-08-01", "2018-08-25")[2]
    print(df)

    errcode, errmsg, df = get_exsymbol_kline("BITFINEX", "btcusdt", "30m", "2017-08-01", "2018-08-25")
    print(str(len(df)), df[0:3])

    errcode, errmsg, result = get_huobi_exchange()
    print(result)

    errcode, errmsg, result = get_huobi_ontime_kline('btcusdt', '15min', '2018-08-18 00:30', '2018-08-23 00:30')
    print(result[0:3])
