import pandas as pd
import tushare as ts
pro = ts.pro_api()
import pdb
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 输入股票名称
name = '华润双鹤'
stocks_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
stock = stocks_list[stocks_list['name'] == name]
ts_code = stock['ts_code'].values[0]

# 当前股价
pice = ts.pro_bar(ts_code=ts_code)[['trade_date', 'close']]
orig_data = pice

# 总股本
total_share = pro.daily_basic(ts_code=ts_code, fields='trade_date,total_share,pe, pb')
orig_data = pd.merge(orig_data, total_share, how='outer', on='trade_date')

# 商誉
goodwill = pro.balancesheet(ts_code=ts_code,  fields='ann_date, goodwill')
goodwill.rename(columns={'ann_date': 'trade_date'}, inplace=True)
orig_data = pd.merge(orig_data, goodwill, how='outer', on='trade_date')

# 基本每股收益, 每股净资产
ps = pro.fina_indicator(ts_code=ts_code,  fields='ann_date, eps, bps')
ps.rename(columns={'ann_date': 'trade_date'}, inplace=True)
orig_data = pd.merge(orig_data, ps, how='outer', on='trade_date')

# 当前股价/基本每股收益
orig_data['指标1'] = orig_data['close'] / orig_data['eps']

# 当前股价/每股净资产
orig_data['指标2'] = orig_data['close'] / orig_data['bps']
orig_data = orig_data.interpolate()


# 绘图
orig_data = orig_data[::-1]
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.plot(orig_data['trade_date'], orig_data['指标1'], label='当前股价/基本每股收益')
ax.xaxis.set_major_locator(LinearLocator(numticks=8))
plt.xticks(rotation=45)
ax.legend()

ax = fig.add_subplot(2, 2, 2)
ax.plot(orig_data['trade_date'], orig_data['指标2'], label='当前股价/每股净资产')
ax.xaxis.set_major_locator(LinearLocator(numticks=8))
plt.xticks(rotation=45)
ax.legend()

ax = fig.add_subplot(2, 2, 3)
ax.plot(orig_data['trade_date'], orig_data['pe'], label='PE')
ax.xaxis.set_major_locator(LinearLocator(numticks=8))
plt.xticks(rotation=45)
ax.legend()

ax = fig.add_subplot(2, 2, 4)
ax.plot(orig_data['trade_date'], orig_data['pb'], label='PB')
ax.xaxis.set_major_locator(LinearLocator(numticks=8))
plt.xticks(rotation=45)
ax.legend()


plt.show()





















