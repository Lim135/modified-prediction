#首先导入需要用到的库
import requests
import pandas as pd
import json
#从网站获取指定股票的历史数据
def get_stock_data(stock_code):
    import requests

    cookies = {
        'qgqp_b_id': 'cbb05b2dbed6d59c9a0b031bd04ea4c6',
        'emshistory': '%5B%22%E8%A3%95%E5%90%8C%E7%A7%91%E6%8A%80%22%5D',
        'HAList': 'ty-1-600598-%u5317%u5927%u8352%2Cty-0-002831-%u88D5%u540C%u79D1%u6280%2Cty-0-000568-%u6CF8%u5DDE%u8001%u7A96%2Cty-0-000858-%u4E94%20%u7CAE%20%u6DB2%2Cty-1-600519-%u8D35%u5DDE%u8305%u53F0%2Cty-0-000061-%u519C%20%u4EA7%20%u54C1%2Cty-0-300059-%u4E1C%u65B9%u8D22%u5BCC',
        'st_si': '68520467292623',
        'st_pvi': '56945317373961',
        'st_sp': '2024-04-20%2011%3A16%3A09',
        'st_inirUrl': 'https%3A%2F%2Fquote.eastmoney.com%2Fsz300059.html',
        'st_sn': '1',
        'st_psi': '20240420195414472-113200301201-8694526986',
        'st_asi': 'delete',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        # 'Cookie': 'qgqp_b_id=cbb05b2dbed6d59c9a0b031bd04ea4c6; emshistory=%5B%22%E8%A3%95%E5%90%8C%E7%A7%91%E6%8A%80%22%5D; HAList=ty-1-600598-%u5317%u5927%u8352%2Cty-0-002831-%u88D5%u540C%u79D1%u6280%2Cty-0-000568-%u6CF8%u5DDE%u8001%u7A96%2Cty-0-000858-%u4E94%20%u7CAE%20%u6DB2%2Cty-1-600519-%u8D35%u5DDE%u8305%u53F0%2Cty-0-000061-%u519C%20%u4EA7%20%u54C1%2Cty-0-300059-%u4E1C%u65B9%u8D22%u5BCC; st_si=68520467292623; st_pvi=56945317373961; st_sp=2024-04-20%2011%3A16%3A09; st_inirUrl=https%3A%2F%2Fquote.eastmoney.com%2Fsz300059.html; st_sn=1; st_psi=20240420195414472-113200301201-8694526986; st_asi=delete',
        'Referer': 'https://quote.eastmoney.com/',
        'Sec-Fetch-Dest': 'script',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    response = requests.get(
        'https://push2his.eastmoney.com/api/qt/stock/kline/get?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&beg=0&end=20500101&ut=fa5fd1943c7b386f172d6893dbfba10b&rtntype=6&secid=1.600598&klt=101&fqt=1&cb=jsonp1713614524312',
        cookies=cookies,
        headers=headers,
    )
    print(response.text)
    return response
def store_data_to_pandas(response):
    rep_text=response.text.replace('jsonp1713614524312(','').replace(');','')
    json_data=json.loads(rep_text)
    stock_data=json_data['data']['klines']
    pd_data=pd.DataFrame(columns=['股票代码','股票名称','交易日期','开盘价','收盘价','最高价','最低价','成交量','成交额','振幅','涨跌幅%','涨跌额'])
    for i in range(len(stock_data)):
        pd_data.loc[i]=[json_data['data']['code'],json_data['data']['name'],stock_data[i].split(',')[0],stock_data[i].split(',')[1],stock_data[i].split(',')[2],stock_data[i].split(',')[3],stock_data[i].split(',')[4],stock_data[i].split(',')[5],stock_data[i].split(',')[6],
                     stock_data[i].split(',')[7],stock_data[i].split(',')[8],stock_data[i].split(',')[9]]
    return pd_data
final_data=pd.DataFrame(columns=['股票代码','股票名称','交易日期','开盘价','收盘价','最高价','最低价','成交量','成交额','振幅','涨跌幅%','涨跌额'])
for i in ['600519']:
    tmp_value=store_data_to_pandas(get_stock_data(i))
    final_data=pd.concat([final_data,tmp_value])
final_data.head()
final_data.to_csv('北大荒stock_data.csv', index=False)
