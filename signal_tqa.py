#!/usr/bin/python
# coding=utf-8
import os
import time
import pandas as pd
import datetime
import requests
import talib


if __name__ == "__main__":
    os.chdir(r'F:\\PycharmProjects\\indicators')
    client = requests.session()
    headers = {'Content-Type': 'application/json', 'Connection': 'keep-alive'}
    s_date = '20170701'
    table_b = 'zf_tqa_b'
    table_s = 'zf_tqa_s'
    today1 = datetime.date.today()
    print(today1)
    transferstr = lambda x: datetime.datetime.strftime(x, '%Y%m%d')
    today = datetime.datetime.strftime(today1, '%Y%m%d')
    print(today)
    s_time = datetime.datetime.strftime(today1, '%Y-%m-%d') + ' 09:30:00'
    s_time = time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))
    e_time = datetime.datetime.strftime(today1, '%Y-%m-%d') + ' 14:45:00'
    e_time = time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))
    calen, firstday, yesterday = get_y_f_calen(s_date, today)
    # today = '20170905'

    if today in calen.day.tolist():
        for i in range(100000):
            t0 = time.time()
            if t0 > e_time:
                break
            else:
                allstarttime = '20170701'
                WIN_ATR = 5
                ATR_n = 2.5
                N1 = 20
                road_period = 10
                N_ATR = 11
                stockcodeNormal = getStockcode('19900101', yesterday)
                stockcodeNormal = stockcodeNormal[stockcodeNormal['LIST_DATE'] < '20160701']
                codelist = stockcodeNormal.stock_code.tolist()
                now_data_df = get_hq_real_time(today)[['stock_code', 'topen', 'thigh', 'tlow', 'tclose', 'date_time']]
                print(now_data_df)
                FAC = get_db_fac(today, today)
                now_fac_data = get_db_fac_data(FAC, now_data_df)\
                    [['stock_code', 'date_time', 'topen', 'thigh', 'tlow', 'tclose', 'open', 'high', 'low', 'close']]
                reindex_data = get_hq_data_zf(allstarttime, yesterday) \
                    [['stock_code', 'date_time', 'topen', 'thigh', 'tlow', 'tclose', 'open', 'high', 'low', 'close']]
                reindex_data = reindex_data.append(now_fac_data)\
                    .assign(close_1=lambda df: df.close.shift(1))\
                    .sort_values(['stock_code', 'date_time'])
                print(reindex_data)
                stockdata = reindex_data.set_index(['stock_code'])
                hq_data = reindex_data.set_index(['stock_code', 'date_time'])
                today_data = stockdata[stockdata['date_time'] == today]
                print(today_data)
                yesday_data = stockdata[stockdata['date_time'] == yesterday]
                today_code = today_data.index.values.tolist()
                csd = MysqlDBConnector('db_xtxg')
                lst = []
                if len(today_data) > 0:
                    today_lst = []
                    for code, group in reindex_data.groupby(['stock_code']):
                        print(code)
                        if len(group) > N1:
                            group['h10'] = talib.MAX(group['high'].values, road_period)
                            group['l10'] = talib.MIN(group['low'].values, road_period)
                            group['ma'] = talib.MA(group['close'].values, N1)
                            group = group.assign(h10=lambda df: df.h10.shift(1)) \
                                .assign(l10=lambda df: df.l10.shift(1)) \
                                .assign(c_ma=lambda df: df.close - df.ma) \
                                .assign(ma_zd=lambda df: df.ma - df.ma.shift(1))\
                                .tail(1)
                            today_lst.append(group)
                    today_df = pd.concat(today_lst)
                    print(today_df)
                    slect_f_df = today_df[(today_df['date_time'] == today) & (today_df['high'] > today_df['h10']) &
                                          (today_df['c_ma'] > 0) & (today_df['ma_zd'] > 0)]
                    print(slect_f_df)
                    sx_stmt = 'select * from factor_stock'
                    sx_poll = csd.get_data_from_query(sx_stmt) \
                        .assign(date_time=lambda df: df.Date.apply(transferstr)) \
                        .assign(stock_code=lambda df: df.StockCode) \
                        .sort_values(['date_time'], ascending=False)

                    sx_poll = sx_poll[sx_poll['date_time'] == max(sx_poll['date_time'])] \
                        .assign(stock_code=lambda df: df.stock_code.apply(lambda x: conver_innercode(x))) \
                        [['stock_code']]
                    print(sx_poll)
                    if len(slect_f_df) > 0:
                        slect_f_df = slect_f_df[['stock_code', 'c_ma']]\
                            .merge(sx_poll, on=['stock_code'])
                        hold_stmt = 'select * from ' + table_s
                        hold = csd.get_data_from_query(hold_stmt)
                        hold = hold[hold['position_flag'] == 0]
                        print(hold)
                        buy_to_sql = []
                        if len(slect_f_df) > 0:
                            for code in slect_f_df.stock_code:
                                if (code not in hold['stock_code'].tolist()) & (code in codelist):
                                    buy_to_sql.append(slect_f_df[slect_f_df['stock_code'] == code]\
                                                      .assign(action_price=today_data.at[code, 'tclose']))
                            if len(buy_to_sql) > 0:
                                buy_to_sql = pd.concat(buy_to_sql)
                                if len(buy_to_sql) > 10 - len(hold):
                                    buy_to_sql = buy_to_sql.head(10 - len(hold))
                                if len(buy_to_sql) > 0:
                                    buy_to_sql = buy_to_sql.sort_values(['c_ma'], ascending=False)[['stock_code', 'action_price']]\
                                        .assign(action_time=today)\
                                        .assign(position_flag=0)\
                                        .assign(action_flag=0)\
                                        .assign(action_type=1)\
                                        .assign(s_position=0)\
                                        .assign(e_position=0.1)
                                    csd.write_data_to_db(buy_to_sql, table_b, mode=3)
                                    lst = []
                                    for idx, col in buy_to_sql.iterrows():
                                        code = col.stock_code
                                        cost_price = today_data.at[code, 'tclose'] / 10000
                                        low_b = 0
                                        high_b = 1000
                                        row_ = []
                                        row_.append(today)
                                        row_.append(0)
                                        row_.append(code)
                                        row_.append(low_b)
                                        row_.append(high_b)
                                        row_.append(0)
                                        row_.append(0)
                                        row_.append(cost_price)
                                        row_.append(0)
                                        row_.append(0.1)
                                        row_.append(today)
                                        lst.append(row_)
                                    s_df = pd.DataFrame(lst,
                                                        columns=['action_time', 'position_flag', 'stock_code', 'lower_bound', 'upper_bound',
                                                                 'action_flag', 'action_type', 'action_price', 's_position', 'e_position',
                                                                 'update_time'])
                                    csd.write_data_to_db(s_df, table_s, mode=3)
            print(time.time() - t0)