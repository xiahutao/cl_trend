#!/usr/bin/python
# coding=utf-8
import socket
import struct
import os
import time
import pandas as pd
import datetime
import numpy as np
import urllib.request, io, sys
import pymssql
import zipfile


def transfer_id(x):
    x = np.str(x)
    if (x[0] == '1') & (x[1] == '6'):
        x = x[1:] + '.sh'
        return x
    elif (x[0] == '2') & (x[1] == '3'):
        x = x[1:] + '.sz'
        return x
    elif (x[0] == '2') & (x[1] == '0'):
        x = x[1:] + '.sz'
        return x
    else:
        return 0


def un_zip(file_name):
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + '_files'):
        pass
    else:
        os.mkdir(file_name + '_files')
    for names in zip_file.namelist():
        zip_file.extract(names, file_name + '_files/')
    zip_file.close()


def downld_gz(date):
    baseurladj = 'http://fintech.jrj.com.cn/l2trade?mk='
    for market in ['sh', 'sz']:
        for minute in ['0930', '1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500']:
            getQueryadj = ''
            getQueryadj = getQueryadj.join((baseurladj, market, '&uz=0&dt=', date, minute))
            name1 = 'dat_data/' + market + '_trade_' + date + '_' + minute + '.dat.gz'
            print(name1)
            r = urllib.request.urlopen(getQueryadj)
            s = r.read()
            with open(name1, "wb") as code:
                code.write(s)


if __name__ == "__main__":
    os.chdir(r'F:\\PycharmProjects\\cl_zjm')
    s_date = '20170301'
    today1 = datetime.date.today()
    print(today1)
    transferstr = lambda x: datetime.datetime.strftime(x, '%Y%m%d')
    today = datetime.datetime.strftime(today1, '%Y%m%d')
    print(today)
    # today = '20180525'
    calen, firstday, yesterday = get_y_f_calen(s_date, today)
    print(calen)
    calen = calen[(calen['day'] >= today) & (calen['day'] <= today)]
    # calen = calen[(calen['day'] >= '20170913') & (calen['day'] <= '20170926')]
    print(calen)

    for date in calen.day:
        downld_gz(date)
