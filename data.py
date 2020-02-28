import requests
import time
import execjs
import matplotlib.pyplot as plt
import json


class Getdata(object):
    def __init__(self):
        self.now = time.strftime("%Y%m%d%H%M%S", time.localtime())

    def getUrl(self, fscode):
        head = 'http://fund.eastmoney.com/pingzhongdata/'
        tail = '.js?v=' + self.now
        return head + fscode + tail

    def getWorth(self, fscode):
        # 用requests获取到对应的文件
        content = requests.get(self.getUrl(fscode))

        # 使用execjs获取到相应的数据
        jsContent = execjs.compile(content.text)
        name = jsContent.eval('fS_name')
        code = jsContent.eval('fS_code')
        # 单位净值走势
        netWorthTrend = jsContent.eval('Data_netWorthTrend')
        # 累计净值走势
        ACWorthTrend = jsContent.eval('Data_ACWorthTrend')

        netWorth = []
        ACWorth = []

        # 提取出里面的净值
        for dayWorth in netWorthTrend[::-1]:
            netWorth.append(dayWorth['y'])

        for dayACWorth in ACWorthTrend[::-1]:
            ACWorth.append(dayACWorth[1])
        return netWorth, ACWorth


if __name__ == '__main__':

    model = Getdata()
    '''
    http://fund.eastmoney.com/
    天弘中证500指数A(000962)
    天弘中证500指数增强C(001557)
    兴全沪深300指数(LOF)A(163407)
    上证指数000001
    '''
    for index in ['000962', '001557', '163407', '000001']:
        plt.figure()
        # 净值
        netWorth, ACWorth = model.getWorth(index)
        days = len(netWorth)
        plt.plot(range(days), netWorth[::-1])

        # 60天
        temp1 = 60
        rollmin = [min(netWorth[i:i+temp1]) for i in range(days - temp1 + 1)]
        plt.plot(range(temp1-1, days), rollmin[::-1], label='60day_min')
        rollmean = [sum(netWorth[i:i + temp1])/temp1 for i in range(days - temp1 + 1)]
        plt.plot(range(temp1 - 1, days), rollmean[::-1], label='60day_mean')

        # 30天最
        temp1 = 30
        rollmin = [min(netWorth[i:i+temp1]) for i in range(days - temp1 + 1)]
        plt.plot(range(temp1-1, days), rollmin[::-1], label='30day_min')
        rollmean = [sum(netWorth[i:i + temp1])/temp1 for i in range(days - temp1 + 1)]
        plt.plot(range(temp1 - 1, days), rollmean[::-1], label='30day_mean')
        plt.axis([days-60, days, min(netWorth[:120])*0.99, max(netWorth[:120])*1.01])
        plt.legend()
    plt.show()