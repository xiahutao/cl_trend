# coding=utf-8
'''
Created on 2018-08-01

@author: fang.zhang
'''


ProductionConnPara = {
    "server": '149.28.94.32',
    "user": 'zhangfang',
    "password": 'qitianDasheng699!',
    "database": 'quantytest',
}


LocalConnPara = {
    "server": '127.0.0.1',
    "user": 'root',
    "password": '1234',
    "database": 'db_test',
}

MssqlConnParaMap = {'production': ProductionConnPara, 'local': LocalConnPara}
