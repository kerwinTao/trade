import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
import pdb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LinearLocator
import datetime
from findN import findN
from findTREND import findTREND
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pd.get_option('display.width')
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
