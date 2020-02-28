import pandas as pd
import tushare as ts
import pdb
pd.get_option('display.width')
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
pro = ts.pro_api()


stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
with open('done.txt', 'r', encoding='UTF-8') as f:
    for line in f.readlines():

        if len(line) > 5:
            val = line.split(' ')[1].split('\n')[0]
            if val in stocks_list['name'].values:
                stocks_list.loc[stocks_list['name'] == val, 'done'] = 1
            else:
                print([val])
stocks_list.to_csv('stocks_list.csv', index=False)