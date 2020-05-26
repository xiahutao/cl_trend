#!/usr/bin/python
# coding=utf-8
import struct
import os
import time
import pandas as pd
import datetime
import numpy as np
import urllib.request, io, sys
import MySQLdb
import pymssql
import sqlalchemy
from sqlalchemy import create_engine
import requests
import talib as ta
import math
import talib
from mssqlDBConnectorPeng import get_hq_data_zf

PGeniusConnPara = {
    "server": '172.16.198.11',
    "user": 'JRJ_Intelligent_Test',
    "password": 'ipFHEu3rKMQkKlobtm56',
    "database": 'PGenius',
}

ProductionConnPara = {
    "server": '10.88.3.90',
    "user": 'intelligent',
    "password": 'in1234',
    "database": 'db_xtxg',
}

dateConnPara = {
    "server": '10.88.3.90',
    "user": 'intelligent',
    "password": 'in1234',
    "database": 'db_data',
}

ProductionConnPara_hqdata = {
    "server": '10.88.3.90',
    "user": 'intelligent',
    "password": 'in1234',
    "database": 'db_genius',
}


MssqlConnParaMap = {'pgenius': PGeniusConnPara, 'db_xtxg': ProductionConnPara,
                    'db_hq': ProductionConnPara_hqdata, 'db_date': dateConnPara}


class MysqlDBConnector(object):
    def __init__(self, dbKey=None):
        if dbKey is None:
            self.connPara = MssqlConnParaMap['local']
        else:
            self.connPara = MssqlConnParaMap[dbKey]
        return

    def build_database_connection(self):
        try:
            conn = MySQLdb.connect(host=self.connPara['server'], user=self.connPara['user'],
                                   passwd=self.connPara['password'], db=self.connPara['database'])
        except MySQLdb.DatabaseError  as e:
            print("Can not connect to server")
        return conn

    def build_alchemy_connection(self):
        try:
            engine = create_engine(
                'mysql+mysqldb://' + self.connPara['user'] + ':' + self.connPara['password'] + '@' + self.connPara[
                    'server'] + ':3306/' + self.connPara['database'])
        except engine.Error  as e:
            print("Can not connect to server")
        return engine

    def write_data_to_db(self, datadf, TableName, mode=1):
        engine = self.build_alchemy_connection()
        if mode == 2:
            datadf.to_sql(TableName, engine, if_exists='replace', index=False, index_label=None, chunksize=None,
                          dtype={'data_type': sqlalchemy.types.String(32), 'last_time': sqlalchemy.TIMESTAMP})
        elif mode == 3:
            datadf.to_sql(TableName, engine, if_exists='append', index=False, index_label=None, chunksize=None,
                          dtype=None)
        else:
            datadf.to_sql(TableName, engine, if_exists='fail', index=False, index_label=None, chunksize=None,
                          dtype=None)

    def get_data_from_query(self, stmt):
        conn = self.build_database_connection()
        df = pd.read_sql(stmt, conn)
        conn.close()
        return df

    def get_query_stmt(self, tableName, colNames, constraints, orderby):
        stmt = 'select '
        for col in colNames:
            stmt = stmt + col + ','
        stmt = stmt[0:len(stmt) - 1]
        stmt = stmt + ' from ' + tableName
        if constraints is None:
            stmt = stmt + ''
        else:
            stmt = stmt + constraints

        if orderby is None:
            return stmt
        else:
            stmt = stmt + ' order by '
            stmt = stmt + orderby
            return stmt

    def get_data(self, tableName, colNames, constraints, orderby):
        assert colNames is not None
        try:
            conn = self.build_database_connection()
            stmt = self.get_query_stmt(tableName, colNames, constraints, orderby)
            cursor = conn.cursor()
            t = time.time()
            cursor.execute(stmt)
            df = pd.DataFrame.from_records(cursor.fetchall())
            if len(df) > 0:
                df.columns = colNames
        except MySQLdb.Error as e:
            conn.rollback()
            message = "SqlServer Error %d: %s" % (e.args[0], e.args[1])
            print(message)
        finally:
            cursor.close()
            conn.close()
        print('time elapsed for this oracle query: ', time.time() - t)
        return df

    def update_data_to_db(self, TableName, chng_cloname, chng_clovalue, cloname_lst, value_lst):
        connect = MySQLdb.connect(host=self.connPara['server'], port=3306, user=self.connPara['user'], passwd=self.connPara['password'], db=self.connPara['database'],
                                  charset='utf8')
        cur = connect.cursor()
        sql1 = 'update ' + TableName + ' set ' + chng_cloname + ' = %s where ('
        sql2 = ''
        for colname in cloname_lst:
            sql2 = sql2 + colname + '= %s)&('
        sql = sql1 + sql2[:-2]
        lst = []
        lst.append(chng_clovalue)
        lst.extend(value_lst)
        cur.execute(sql, lst)
        connect.commit()
        connect.close()


class MssqlDBConnector(object):
    def __init__(self, dbKey=None):
        if dbKey is None:
            self.connPara = MssqlConnParaMap['pgenius']
        else:
            self.connPara = MssqlConnParaMap[dbKey]
        return

    def build_database_connection(self):
        try:
            conn = pymssql.connect(self.connPara['server'], self.connPara['user'], self.connPara['password'],
                                   self.connPara['database'])
        except pymssql.DatabaseError  as e:
            print("Can not connect to server")
        return conn

    def get_data_from_query(self, stmt):
        conn = self.build_database_connection()
        df = pd.read_sql(stmt, conn)
        conn.close()
        return df

    def get_query_stmt(self, tableName, colNames, constraints, orderby):
        stmt = 'select '
        for col in colNames:
            stmt = stmt + col + ','
        stmt = stmt[0:len(stmt) - 1]
        stmt = stmt + ' from ' + tableName
        if constraints is None:
            stmt = stmt + ''
        else:
            stmt = stmt + constraints

        if orderby is None:
            return stmt
        else:
            stmt = stmt + ' order by '
            stmt = stmt + orderby
            return stmt

    def get_data(self, tableName, colNames, constraints, orderby):
        assert colNames is not None
        try:
            conn = self.build_database_connection()
            stmt = self.get_query_stmt(tableName, colNames, constraints, orderby)
            cursor = conn.cursor()
            t0 = time.time()
            cursor.execute(stmt)
            df = pd.DataFrame.from_records(cursor.fetchall())
            if len(df) > 0:
                df.columns = colNames
        except pymssql.Error as e:
            conn.rollback()
            message = "SqlServer Error %d: %s" % (e.args[0], e.args[1])
            print(message)
        finally:
            cursor.close()
            conn.close()
        print('time elapsed for this oracle query: ', time.time() - t0)
        return df


def transfer_id(x):
    x = np.str(x)
    if (x[0] == '1') & (x[1] == '6'):
        x = 'sh' + x[1:]
        return x
    elif (x[0] == '2') & (x[1] == '3'):
        x = 'sz' + x[1:]
        return x
    elif (x[0] == '2') & (x[1] == '0'):
        x = 'sz' + x[1:]
        return x
    else:
        return 0


# ==================================================取股票代码
def getStockcode(startDate, endDate):
    csd = MssqlDBConnector('pgenius')
    testCols2 = ['INNER_CODE', 'STOCKCODE', 'LIST_DATE']
    tableName = 'PGenius.dbo.STK_CODE'
    orderby = 'INNER_CODE ASC'
    const1 = ' where STOCKCODE like ' + '\'' + '[036]%' + '\'' + 'and' + ' STATUS_TYPE = ' + '\'' + '正常上市' + '\''
    data = csd.get_data(tableName, testCols2, const1, orderby)
    data = data[(data['LIST_DATE'] <= endDate) & (data['LIST_DATE'] >= startDate)]
    data['STOCKCODE'] = data['STOCKCODE'].apply(lambda x: conver_innercode(x))
    data = data.rename(columns={'STOCKCODE': 'stock_code'})
    return data


def get_stock_mktcap(stockcodeNormal, startDate, endDate):
    csd = MssqlDBConnector('pgenius')
    testCols2 = ['INNER_CODE', 'ENDDATE', 'MKTCAP_1']
    tableName = 'PGenius.dbo.ANA_STK_EXPR_IDX'
    orderby = 'ENDDATE ASC,INNER_CODE ASC'  # ASC指升序,DESC指降序
    const1 = ' where INNER_CODE in ('
    if stockcodeNormal is not None:
        for innercode in stockcodeNormal['INNER_CODE']:
            const1 = const1 + '\'' + str(innercode) + '\'' + ','
    const1 = const1[:-1] + ') and '
    const2 = 'ENDDATE between ' + '\'' + startDate + '\'' + ' and ' + '\'' + endDate + '\''
    constraints = const1 + const2
    data = csd.get_data(tableName, testCols2, constraints, orderby)\
               .rename(columns={'STOCKCODE': 'stock_code'})\
               .merge(stockcodeNormal, on=['INNER_CODE'])[['stock_code', 'MKTCAP_1']] \
               .sort_values(by='MKTCAP_1') \
               .loc[:, ['stock_code', 'MKTCAP_1']]
    return data


def get_hq_data(s_date, e_date):
    hqstmt = 'select * from tb_stock_data where date between ' + '\'' + s_date + '\'' + ' and ' + '\'' + e_date + '\''
    stockdata = []
    csd_hq = MysqlDBConnector('db_hq')
    data = csd_hq.get_data_from_query(hqstmt) \
        [['date', 'code', 'tclose', 'thigh', 'tlow', 'topen', 'fac']]
    stockdata.append(data[(data['code'] >= 1600000) & (data['code'] <= 1603999)])
    stockdata.append(data[(data['code'] >= 2000000) & (data['code'] <= 2002999)])
    stockdata.append(data[(data['code'] >= 2300000) & (data['code'] <= 2301999)])
    stockdata = pd.concat(stockdata) \
        .assign(date_time=lambda df: df.date.apply(lambda x: str(x))) \
        .assign(tclose=lambda df: df.tclose.apply(lambda x: ('%.2f' % (x / 10000)))) \
        .assign(thigh=lambda df: df.thigh.apply(lambda x: ('%.2f' % (x / 10000)))) \
        .assign(tlow=lambda df: df.tlow.apply(lambda x: ('%.2f' % (x / 10000)))) \
        .assign(topen=lambda df: df.topen.apply(lambda x: ('%.2f' % (x / 10000)))) \
        .assign(topen=lambda df: df.topen.apply(lambda x: float(x))) \
        .assign(thigh=lambda df: df.thigh.apply(lambda x: float(x))) \
        .assign(tlow=lambda df: df.tlow.apply(lambda x: float(x))) \
        .assign(tclose=lambda df: df.tclose.apply(lambda x: float(x))) \
        .assign(open=lambda df: df.topen * df.fac) \
        .assign(high=lambda df: df.thigh * df.fac) \
        .assign(low=lambda df: df.tlow * df.fac) \
        .assign(close=lambda df: df.tclose * df.fac) \
        .assign(stock_code=lambda df: df.code.apply(lambda x: conver_innercode(str(x)[1:]))) \
        .sort_values(by=['stock_code', 'date_time'])
    return stockdata


def get_calen(datedf, s_date, e_date):
    datedf = datedf[(datedf['market'] == 1) & (datedf['openclose'] == 1) & (datedf['day'] <= int(e_date))]
    calen = datedf[(datedf['day'] >= int(s_date)) & (datedf['day'] <= int(e_date))] \
        .assign(day=lambda df: df.day.apply(lambda x: str(x)))
    return calen


def get_h_l(hq, starttime, endtime, N, win_atr, loss_atr):
    hq = hq[(hq['date_time'] >= starttime) & (hq['date_time'] <= endtime)]\
        [['stock_code', 'date_time', 'close', 'open', 'high', 'low']]

    lst = []
    for code, stockonedata in hq.groupby('stock_code'):
        print(code)
        stockonedata = stockonedata\
            .sort_values(['date_time'])\
            .assign(close_1=lambda df: df.close.shift(1))
        stockonedata['atr'] = ta.ATR(stockonedata.high.values, stockonedata.low.values, stockonedata.close.values, N)
        stockonedata = stockonedata\
            .assign(upper_bound=lambda df: (1 + df.atr * win_atr / df.close))\
            .assign(lower_bound=lambda df: (1 - df.atr * loss_atr / df.close))\
            [['stock_code', 'date_time', 'upper_bound', 'lower_bound', 'close_1']]
        stockonedata = stockonedata[stockonedata['date_time'] == endtime]
        if len(stockonedata) > 0:
            lst.append(stockonedata)
    ret = pd.concat(lst)
    return ret


def get_u_l(code_hq, s_time, e_time, N, win_atr, loss_atr):
    print(code_hq)
    code_hq = code_hq.sort_values(['date_time']) \
        .reset_index(drop=False)\
        .set_index(['date_time'])
    code_hq['atr'] = ta.ATR(code_hq.high.values, code_hq.low.values, code_hq.close.values, N)
    hold_data = code_hq['high'][s_time: e_time]
    max_hold_data = hold_data.max()
    print(code_hq['high'][s_time: e_time])
    print(max_hold_data)
    upper = (max_hold_data + code_hq.atr.tolist()[-1] * win_atr) / code_hq.close.tolist()[-1]
    lower = (max_hold_data - code_hq.atr.tolist()[-1] * loss_atr) / code_hq.close.tolist()[-1]
    close_1 = code_hq.close.tolist()[-2]
    return upper, lower, close_1, len(hold_data)


def get_y_f_calen(allstarttime, today):
    csddate = MysqlDBConnector('db_date')
    datestmt = 'select * from tb_holiday'
    datedf = csddate.get_data_from_query(datestmt)
    calen = get_calen(datedf, allstarttime, today)
    firstday = calen.day.values[-3]
    yesterday = calen.day.values[-2]
    return calen, firstday, yesterday


def maxRetrace(list):
    row = []
    endN = 0
    starN = 0
    for i in range(len(list)):
        row.append(1 - list[i] / max(list[:i + 1]))
        if 1 - list[i] / max(list[:i + 1]) == max(row):
            endN = i
    for n in range(endN):
        if list[n] == max(list[:n + 1]):
            starN = n
    Max = max(row)
    return Max, starN, endN


def conver_innercode(code):
    if code[0] == '6':
        code = 'sh' + str(code)[:6]
    elif code[0] == '0' or code[0] == '3':
        code = 'sz' + str(code)[:6]
    return code


def getOneWebPrices(Ticker):
    baseUrl = 'http://fintech.jrj.com.cn/hq?cmd=getStock&market='
    getQuery = ''
    getQuery = getQuery.join((baseUrl, Ticker[:2], '&code=', Ticker[2:]))
    response = client.get(getQuery, headers=headers)
    jsondata = response.json()
    pricedf = pd.DataFrame(jsondata)
    if len(pricedf) > 0:
        ret = pricedf.loc[0:0, ['nSecurityID', 'llVolume', 'nOpenPx', 'nHighPx', 'nLowPx', 'nLastPx', 'nPreClosePx']]
        return ret
    else:
        return []


if __name__ == "__main__":
    os.chdir(r'F:\\PycharmProjects\\indicators')
    client = requests.session()
    headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive'}
    t0 = time.time()
    table_s_lst = ['zf_tqa_s', 'zf_muti_head_s']
    s_date = '20170701'
    LOSS_ATR = 2.5
    N_ATR = 11
    WIN_ATR = 5
    today1 = datetime.date.today()
    print(today1)
    transferstr = lambda x: datetime.datetime.strftime(x, '%Y%m%d')
    today = datetime.datetime.strftime(today1, '%Y%m%d')
    print(today)
    calen, firstday, yesterday = get_y_f_calen(s_date, today)
    #today = '20170905'
    if today in calen.day.tolist():
        allstarttime = '20170701'
        stockcodeNormal = getStockcode('19900101', yesterday)
        stockcodeNormal = stockcodeNormal[stockcodeNormal['LIST_DATE'] < '20160701']
        mktcapdf = get_stock_mktcap(stockcodeNormal, firstday, firstday)
        codelist = stockcodeNormal.stock_code.tolist()

        reindex_data = get_hq_data_zf(allstarttime, today) \
            [['stock_code', 'date_time', 'topen', 'thigh', 'tlow', 'tclose', 'open', 'high', 'low', 'close']]
        stockdata = reindex_data.set_index(['stock_code'])
        hq_data = reindex_data.set_index(['stock_code', 'date_time'])
        today_data = stockdata[stockdata['date_time'] == today]
        print(today_data)
        yesday_data = stockdata[stockdata['date_time'] == yesterday]
        today_code = today_data.index.values.tolist()
        csd = MysqlDBConnector('db_xtxg')
        lst = []
        if len(today_data) > 0:
            for table_s in table_s_lst:
                trans_updatetime_to_str = lambda x: str(x)[:4] + str(x)[5:7] + str(x)[8:10]
                hold_stmt = 'select * from ' + table_s
                hold_all = csd.get_data_from_query(hold_stmt)
                len_s = len(hold_all)
                hold = hold_all[(hold_all['position_flag'] == 0)]
                print(hold)

                if len(hold) > 0:
                    hold = hold.assign(s_time=lambda df: df.update_time.apply(trans_updatetime_to_str))\
                        .assign(action_time=lambda df: df.action_time.apply(transferstr))
                print(hold)
                if len(hold) > 0:
                    for idx, col in hold.iterrows():

                        code = col.stock_code
                        print(code)
                        low_b = float(col.lower_bound)
                        high_b = float(col.upper_bound)

                        csd.update_data_to_db(table_s, 'action_time', today, ['stock_code', 'position_flag'], [code, 0])
                        if code in today_code:
                            cost = hq_data.loc[code, 'open'][col.s_time]
                            upper, lower, close_1, hold_days = get_u_l(stockdata.loc[code], col.s_time, col.action_time, N_ATR, WIN_ATR, LOSS_ATR)
                            if hold_days > 1:
                                if today_data.at[code, 'high']/close_1 > high_b:
                                    sell = close_1 * high_b
                                    action_sell = sell * today_data.at[code, 'tclose'] / today_data.at[code, 'close']
                                    csd.update_data_to_db(table_s, 'action_price', math.floor(100 * action_sell) / 100, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 's_position', 0.1, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'e_position', 0, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'position_flag', 1, ['stock_code', 'position_flag'], [code, 0])

                                elif today_data.at[code, 'open']/close_1 < low_b:
                                    sell = today_data.at[code, 'open']
                                    action_sell = today_data.at[code, 'topen']
                                    csd.update_data_to_db(table_s, 'action_price', math.floor(100 * action_sell) / 100,
                                                          ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 's_position', 0.1, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'e_position', 0, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'position_flag', 1, ['stock_code', 'position_flag'], [code, 0])

                                elif today_data.at[code, 'low']/close_1 < low_b:
                                    sell = close_1 * low_b
                                    action_sell = sell * today_data.at[code, 'tclose'] / today_data.at[code, 'close']
                                    csd.update_data_to_db(table_s, 'action_price', math.floor(100 * action_sell) / 100,
                                                          ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 's_position', 0.1, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'e_position', 0, ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'position_flag', 1, ['stock_code', 'position_flag'], [code, 0])

                                else:
                                    csd.update_data_to_db(table_s, 'action_price', math.floor(100 * today_data.at[code, 'tclose'] ) / 100,
                                                          ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'lower_bound', lower,
                                                          ['stock_code', 'position_flag'], [code, 0])
                                    csd.update_data_to_db(table_s, 'upper_bound', upper,
                                                          ['stock_code', 'position_flag'], [code, 0])

                            else:
                                csd.update_data_to_db(table_s, 'action_price', math.floor(100 * today_data.at[code, 'tclose'] ) / 100,
                                                      ['stock_code', 'position_flag'], [code, 0])

                                csd.update_data_to_db(table_s, 'lower_bound', lower,
                                                      ['stock_code', 'position_flag'], [code, 0])
                                csd.update_data_to_db(table_s, 'upper_bound', upper,
                                                      ['stock_code', 'position_flag'], [code, 0])