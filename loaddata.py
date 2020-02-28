import pandas as pd
import tushare as ts
import pdb
from tqdm import tqdm
import os
pd.get_option('display.width')
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
pro = ts.pro_api()

#  读取数据
stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
for i, item in tqdm(stocks_list.iterrows()):
    name = item['name']
    ts_code = item['ts_code']
    df = ts.pro_bar(ts_code=ts_code, adj='qfq')
    filename = name.replace('*', 'xing') + '.csv'
    path = os.path.join('.\\stockdata\\', filename)
    df.to_csv(path, index=False)
