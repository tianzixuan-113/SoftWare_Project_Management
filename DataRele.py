import baostock as bs
import pandas as pd
import numpy as np
import csv
import re
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from collections import deque

def DailyDataget(codes,target,startdate,enddate):
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    #l=['600036','601318','600104','600019','600895','600519','600276','600900','600028','600536','600745','600887']
    for idx,code in enumerate(codes,0): 
    
        rs = bs.query_history_k_data_plus(f"sh.{code}",
            "date,open,high,low,close,volume,amount,turn,preclose,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=startdate, end_date=enddate,
            frequency="d", adjustflag="3")
        print('query_history_k_data_plus respond error_code:'+rs.error_code)
        print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        #### 结果集输出到csv文件 ####   
        result.to_csv(f"{target}/_{idx}.csv", index=False)

    #### 登出系统 ####
    bs.logout()

def CompDataget(codes,target,startyear,endyear):
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    for idx,code in enumerate(codes,0): 
        # 查询季频估值指标盈利能力
        # data_list_save = []
        # data_list_borrow = []
        # data_list_rr = []
        # data_list_offer = []
        # data_list_left = []

        profit_list = []
        operation_list = []
        growth_list = []
        balance_list = []
        cash_flow_list = []
        dupont_list = []
        
        for x in range(startyear,endyear):
            # save = bs.query_deposit_rate_data(start_date=f"{x}-01-01", end_date=f"{x}-12-31")
            # while (save.error_code == '0') & save.next():
            #     data_list_save.append(save.get_row_data())
                
            # borrow = bs.query_loan_rate_data(start_date=f"{x}-01-01", end_date=f"{x}-12-31")
            # while (borrow.error_code == '0') & borrow.next():
            #     data_list_borrow.append(borrow.get_row_data())
                
            # req_res = bs.query_required_reserve_ratio_data(start_date=f"{x}-01-01", end_date=f"{x}-12-31")
            # while (req_res.error_code == '0') & req_res.next():
            #     data_list_rr.append(req_res.get_row_data())
                
            # offer = bs.query_money_supply_data_month(start_date=f"{x}-01", end_date=f"{x}-12")
            # while (offer.error_code == '0') & offer.next():
            #     data_list_offer.append(offer.get_row_data())
                
            # left = bs.query_money_supply_data_year(start_date=f"{x}", end_date=f"{x}")
            # while (left.error_code == '0') & left.next():
            #     data_list_left.append(left.get_row_data())
                
            for y in range(1,5):
                rs_profit = bs.query_profit_data(code=code, year=x, quarter=y)
                while (rs_profit.error_code == '0') & rs_profit.next():
                    profit_list.append(rs_profit.get_row_data())
                    
                rs_operation = bs.query_operation_data(code="sh.600206", year=x, quarter=y)
                while (rs_operation.error_code == '0') & rs_operation.next():
                    operation_list.append(rs_operation.get_row_data())
                    
                rs_growth = bs.query_growth_data(code="sh.600206", year=x, quarter=y)
                while (rs_growth.error_code == '0') & rs_growth.next():
                    growth_list.append(rs_growth.get_row_data())
                    
                rs_balance = bs.query_balance_data(code="sh.600206", year=x, quarter=y)
                while (rs_balance.error_code == '0') & rs_balance.next():
                    balance_list.append(rs_balance.get_row_data())
                    
                rs_cash_flow = bs.query_cash_flow_data(code="sh.600206", year=x, quarter=y)
                while (rs_cash_flow.error_code == '0') & rs_cash_flow.next():
                    cash_flow_list.append(rs_cash_flow.get_row_data())
                    
                rs_dupont = bs.query_dupont_data(code="sh.600206", year=x, quarter=y)
                while (rs_dupont.error_code == '0') & rs_dupont.next():
                    dupont_list.append(rs_dupont.get_row_data())

        # re_save = create_df(data_list_save, save.fields)
        # re_borrow= create_df(data_list_borrow, borrow.fields)
        # rr = create_df(data_list_rr, req_res.fields)
        # re_offer = create_df(data_list_offer, offer.fields)
        # re_left = create_df(data_list_left, left.fields)

        result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
        result_profit = result_profit.replace('', np.nan)

        result_operation = pd.DataFrame(operation_list, columns=rs_operation.fields)
        result_operation = result_operation.replace('', np.nan)

        result_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)
        result_growth = result_growth.replace('', np.nan)

        result_balance = pd.DataFrame(balance_list, columns=rs_balance.fields)
        result_balance = result_balance.replace('', np.nan)

        result_cash_flow = pd.DataFrame(cash_flow_list, columns=rs_cash_flow.fields)
        result_cash_flow = result_cash_flow.replace('', np.nan)

        result_dupont = pd.DataFrame(dupont_list, columns=rs_dupont.fields)
        result_dupont = result_dupont.replace('', np.nan)

        # 合并所有DataFrame
        merged_df = result_profit.copy()

        for df in [result_operation, result_growth, result_balance, 
                result_cash_flow, result_dupont]:
            merged_df = pd.merge(merged_df, df, on=['code','pubDate','statDate'], how='outer')
            
        # 空值处理 - 全部用0填充
        filled_df = merged_df.fillna(0)

        filled_df.to_csv(f'{target}/_{idx}.csv', index=False)

    
def betterCsv(inpath, originalyear='2000'):
    before= 0
    dist={'01':-1,
          '02':30,
          '03':59,
          '04':90,
          '05':120,
          '06':151,
          '07':181,
          '08':212,
          '09':243,
          '10':273,
          '11':304,
          '12':334,
          }
    for path in inpath:
        with open(f'{path}.csv',"r+",encoding='utf-8',newline='') as d,open(f"{path}_better.csv", "w", encoding="utf-8", newline="") as f:
            csv_reader = csv.reader(d)
            csv_writer = csv.writer(f)
            #name=['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM']
            #csv_writer.writerow(name)
            
            #最大60日
            last=[]
            zlast=[]
            #open：0
            #high：1
            #low：2
            #close：3
            data=[[],[],[],[]]
            z=[]
            
            for id,row in enumerate(csv_reader,1):
                #输入行标        
                if(row[1]=='open'):
                    z=['date',row[1],row[2],row[3],row[4],
                       'open_month_volatility_21','open_month_volatility_32','open_month_volatility_65','open_month_volatility_96',
                       'high_month_volatility_21','high_month_volatility_32','high_month_volatility_65','high_month_volatility_96',
                       'low_month_volatility_21','low_month_volatility_32','low_month_volatility_65','low_month_volatility_96',
                       'close_month_volatility_21','close_month_volatility_32','close_month_volatility_65','close_month_volatility_96',
                        'open_year_volatility_21','open_year_volatility_32','open_year_volatility_65','open_year_volatility_96','open_year_volatility_127','open_year_volatility_190','open_year_volatility_252',
                        'high_year_volatility_21','high_year_volatility_32','high_year_volatility_65','high_year_volatility_96','high_year_volatility_127','high_year_volatility_190','high_year_volatility_252',
                        'low_year_volatility_21','low_year_volatility_32','low_year_volatility_65','low_year_volatility_96','low_year_volatility_127','low_year_volatility_190','low_year_volatility_252',
                        'close_year_volatility_21','close_year_volatility_32','close_year_volatility_65','close_year_volatility_96','close_year_volatility_127','close_year_volatility_190','close_year_volatility_252',
                        'open_ma_5','high_ma_5','low_ma_5','close_ma_5',
                        'open_ma_10','high_ma_10','low_ma_10','close_ma_10',
                        'open_ma_20','high_ma_20','low_ma_20','close_ma_20',
                        'open_ma_30','high_ma_30','low_ma_30','close_ma_30',
                        'open_ma_60','high_ma_60','low_ma_60','close_ma_60',
                        'open_logma_5','high_logma_5','low_logma_5','close_logma_5',
                        'open_logma_10','high_logma_10','low_logma_10','close_logma_10',
                        'open_logma_20','high_logma_20','low_logma_20','close_logma_20',
                        'open_logma_30','high_logma_30','low_logma_30','close_logma_30',
                        'open_logma_60','high_logma_60','low_logma_60','close_logma_60',
                        'open_short_EMA','high_short_EMA','low_short_EMA','close_short_EMA',
                        'open_long_EMA','high_long_EMA','low_long_EMA','close_long_EMA',
                        'open_DIF','high_DIF','low_DIF','close_DIF',
                        'open_DEA','high_DEA','low_DEA','close_DEA',
                        'open_MACD','high_MACD','low_MACD','close_MACD',
                        row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],
                        'K_value','D_value','J_value']
                    
                    csv_writer.writerow(z)
                    continue
                
                searchObj = re.search( r'(.*)-(.*)-(.*)', row[0], re.M|re.I)
                if(not searchObj.group(1)==originalyear):
                    before+=1
                    originalyear=searchObj.group(1)
                zz=dist[searchObj.group(2)]+int(searchObj.group(3))+366*before
                
                #初始化row，去空值
                for idx,d in enumerate(row,0):
                    if(d==''):
                        row[idx]=0
                    if(idx>0):
                        row[idx]=float(row[idx])
                        
                #初始化zlast,data
                if(zlast==[]):
                    zlast=[zz,
                           row[1],row[2],row[3],row[4],
                           0,0,0,0,
                           0,0,0,0,
                           0,0,0,0,
                           0,0,0,0,
                           0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           row[1],row[2],row[3],row[4],
                           0,0,0,0,
                           0,0,0,0,
                           0,0,0,0,
                           row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],
                           50,50,50]
                    for x in range(4):
                        data[x].append(row[x+1])
                                                    
                
                        
                #初始化z
                z=[zz,row[1],row[2],row[3],row[4]]
                z0=[row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]]
                
                #数据加入对应列表
                for x in range(4):
                    data[x].append(row[x+1])
                
                #删除多余数据
                if(id>252):
                    for x in range(4):
                        data[x].pop(0)
                        
                #计算数据
                #计算年、月波动率
                #年：21,32,65,96,127,190,252
                #月：21,32,65,96
                year_volatility=[]
                month_volatility=[]
                for x in range(4):
                    log_return_21=np.diff(np.log(data[x][-22:]))
                    log_return_32=np.diff(np.log(data[x][-33:]))
                    log_return_65=np.diff(np.log(data[x][-66:]))
                    log_return_96=np.diff(np.log(data[x][-97:]))
                    log_return_127=np.diff(np.log(data[x][-128:]))
                    log_return_190=np.diff(np.log(data[x][-191:]))
                    log_return_252=np.diff(np.log(data[x][-253:]))
                    
                    month_volatility.append(log_return_21.std()*np.sqrt(12))
                    month_volatility.append(log_return_32.std()*np.sqrt(12))
                    month_volatility.append(log_return_65.std()*np.sqrt(12))
                    month_volatility.append(log_return_96.std()*np.sqrt(12))
                    
                    year_volatility.append(log_return_21.std()*np.sqrt(250))
                    year_volatility.append(log_return_32.std()*np.sqrt(250))
                    year_volatility.append(log_return_65.std()*np.sqrt(250))
                    year_volatility.append(log_return_96.std()*np.sqrt(250))
                    year_volatility.append(log_return_127.std()*np.sqrt(250))
                    year_volatility.append(log_return_190.std()*np.sqrt(250))
                    year_volatility.append(log_return_252.std()*np.sqrt(250))
                    
                    
                
                #计算均线（简单、指数）：5日，10日，20日，30日，60日
                ma_5=[]
                ma_10=[]
                ma_20=[]
                ma_30=[]
                ma_60=[]
                logma_5=[]
                logma_10=[]
                logma_20=[]
                logma_30=[]
                logma_60=[]
                for x in range(4):
                    ma_5.append(np.mean(data[x][-5:]))
                    ma_10.append(np.mean(data[x][-10:]))
                    ma_20.append(np.mean(data[x][-20:]))
                    ma_30.append(np.mean(data[x][-30:]))
                    ma_60.append(np.mean(data[x][-60:]))
                    #w_l=5 if(len(data[x])>=5) else len(data[x])
                    #w=np.exp(np.linspace(0,1,w_l))
                    #logma_10.append(np.average(data[x][-5:],weights=w))
                    logma_5.append((zlast[69+x]*3+data[x][-1]*2)/5)
                    #w_l=10 if(len(data[x])>=10) else len(data[x])
                    #w=np.exp(np.linspace(0,1,w_l))
                    #logma_10.append(np.average(data[x][-10:],weights=w))
                    logma_10.append((zlast[73+x]*8+data[x][-1]*2)/10)
                    #w_l=20 if(len(data[x])>=20) else len(data[x])
                    #w=np.exp(np.linspace(0,1,w_l))
                    #logma_20.append(np.average(data[x][-20:],weights=w))
                    logma_20.append((zlast[77+x]*18+data[x][-1]*2)/20)
                    #w_l=30 if(len(data[x])>=30) else len(data[x])
                    #w=np.exp(np.linspace(0,1,w_l))
                    #logma_30.append(np.average(data[x][-30:],weights=w))
                    logma_30.append((zlast[81+x]*28+data[x][-1]*2)/30)
                    #w_l=60 if(len(data[x])>=60) else len(data[x])
                    #w=np.exp(np.linspace(0,1,w_l))
                    #logma_60.append(np.average(data[x][-60:],weights=w))
                    logma_60.append((zlast[85+x]*58+data[x][-1]*2)/60)
                                    
                #计算MACD
                short_EMA=[]
                long_EMA=[]
                dif=[]
                dea=[]
                macd=[]
                for x in range(4):
                    a1=(zlast[-32+x]*11+data[x][-1]*2)/13
                    short_EMA.append(a1)
                    a2=(zlast[-28+x]*25+data[x][-1]*2)/27
                    long_EMA.append(a2)
                    dif.append(a1-a2)
                    a3=(zlast[-20+x]*7+(a1-a2)*2)/9
                    dea.append(a3)
                    macd.append((a1-a2-a3)*2)                    
                
                #计算KDJ
                if((max(data[1][-9:])-min(data[2][-9:]))==0):
                    rsv=zlast[-3]
                else:
                    rsv=(data[3][-1]-min(data[2][-9:]))/(max(data[1][-9:])-min(data[2][-9:]))*100
                _k=(zlast[-3]*2+rsv)/3
                _d=(zlast[-2]*2+_k)/3
                _j=_k*3-_d*2
                
                #拼合z
                z.extend(month_volatility)
                z.extend(year_volatility)   
                z.extend(ma_5)   
                z.extend(ma_10)   
                z.extend(ma_20)   
                z.extend(ma_30)
                z.extend(ma_60)
                z.extend(logma_5)
                z.extend(logma_10)
                z.extend(logma_20)
                z.extend(logma_30)
                z.extend(logma_60)
                z.extend(short_EMA)
                z.extend(long_EMA)
                z.extend(dif)
                z.extend(dea)
                z.extend(macd)
                z.extend(z0)
                z.extend([_k])
                z.extend([_d])
                z.extend([_j])
                
                #写入z              
                csv_writer.writerow(z)
                last=row
                zlast=z
                
                if(id==2):
                    for x in range(4):
                        data[x].pop(0)
                        
class IncrementalNormalizer:
    """处理多维数据的增量归一化器"""
    
    def __init__(self, dim, epsilon=1e-8):
        """
        初始化多维增量归一化器
        
        参数:
            dim: 数据维度
            epsilon: 防止除零错误的小常数
        """
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)  # 每维度的平方偏差总和
        self.epsilon = epsilon
        
    def update(self, x):
        """
        使用新数据点更新归一化参数
        
        参数:
            x: 形状为 (dim,) 的 numpy 数组，表示一个多维数据点
        """
        if x.shape != (self.dim,):
            raise ValueError(f"输入数据维度应为 ({self.dim},)，但得到 {x.shape}")
            
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    def normalize(self, x):
        """
        使用当前参数对新数据点进行归一化
        
        参数:
            x: 要归一化的多维数据点
            
        返回:
            归一化后的多维数据点
        """
        if self.count < 2:
            return x
        variance = self.M2 / (self.count - 1)
        std_dev = np.sqrt(variance + self.epsilon)
        return (x - self.mean) / std_dev
    
    def get_params(self):
        """获取当前的归一化参数"""
        if self.count < 2:
            return self.mean, np.ones(self.dim)
        variance = self.M2 / (self.count - 1)
        std_dev = np.sqrt(variance + self.epsilon)
        return self.mean, std_dev

class AdaptiveIncrementalNorm:
    """
    结合滑动窗口和动态遗忘因子的自适应增量归一化类
    处理具有m个特征的一维数组流
    """
    def __init__(self, num_features, window_size=100, initial_forget_factor=0.9, 
                 drift_threshold=0.1, min_forget_factor=0.7, max_forget_factor=0.99):
        """
        初始化增量归一化器
        
        参数:
            num_features: 特征数量(m)
            window_size: 滑动窗口大小
            initial_forget_factor: 初始遗忘因子
            drift_threshold: 概念漂移检测阈值
            min_forget_factor: 最小遗忘因子
            max_forget_factor: 最大遗忘因子
        """
        self.num_features = num_features
        self.window_size = window_size
        self.forget_factor = initial_forget_factor
        self.drift_threshold = drift_threshold
        self.min_forget_factor = min_forget_factor
        self.max_forget_factor = max_forget_factor
        
        # 初始化统计量
        self.count = 0
        self.ema_mean = torch.zeros(num_features)
        self.ema_var = torch.zeros(num_features)
        self.window = deque(maxlen=window_size)
        
        # 概念漂移检测
        self.last_mean = torch.zeros(num_features)
        self.drift_detected = False
        
    def update(self, x):
        """
        更新归一化统计量
        
        参数:
            x: 一维张量，形状为[num_features]
        """
        # 确保输入是正确形状的张量
        x = torch.as_tensor(x, dtype=torch.float32)
        assert x.shape == (self.num_features,), f"输入形状应为({self.num_features},), 但得到{tuple(x.shape)}"
        
        # 添加到滑动窗口
        self.window.append(x)
        self.count += 1
        
        # 初始化或更新EMA统计量
        if self.count == 1:
            self.ema_mean = x.clone()
            self.ema_var = torch.zeros_like(x)
            self.last_mean = x.clone()
        else:
            # 检测概念漂移
            if self.count % self.window_size == 0:
                current_window = torch.stack(list(self.window))
                window_mean = current_window.mean(dim=0)
                mean_diff = torch.norm(window_mean - self.last_mean) / (torch.norm(self.last_mean) + 1e-8)
                
                if mean_diff > self.drift_threshold:
                    self.drift_detected = True
                    # 发生漂移时减小遗忘因子，更快适应新分布
                    self.forget_factor = max(self.min_forget_factor, self.forget_factor * 0.95)
                else:
                    self.drift_detected = False
                    # 稳定时增加遗忘因子，信任历史数据
                    self.forget_factor = min(self.max_forget_factor, self.forget_factor * 1.01)
                
                self.last_mean = window_mean.clone()
            
            # 更新EMA统计量
            diff = x - self.ema_mean
            self.ema_mean = self.forget_factor * self.ema_mean + (1 - self.forget_factor) * x
            self.ema_var = self.forget_factor * self.ema_var + (1 - self.forget_factor) * diff * diff
    
    def normalize(self, x):
        """
        使用当前统计量对输入进行归一化
        
        参数:
            x: 一维张量，形状为[num_features]
        
        返回:
            归一化后的张量
        """
        # 确保输入是正确形状的张量
        x = torch.as_tensor(x, dtype=torch.float32)
        
        # 处理方差为零的情况
        std = torch.sqrt(self.ema_var + 1e-8)
        
        # 归一化
        return (x - self.ema_mean) / std
    
    def update_and_normalize(self, x):
        """
        更新统计量并对输入进行归一化
        
        参数:
            x: 一维张量，形状为[num_features]
        
        返回:
            归一化后的张量
        """
        self.update(x)
        return self.normalize(x)
    
    def get_current_stats(self):
        """返回当前的统计量"""
        return {
            'ema_mean': self.ema_mean,
            'ema_std': torch.sqrt(self.ema_var + 1e-8),
            'forget_factor': self.forget_factor,
            'drift_detected': self.drift_detected,
            'window_size': len(self.window),
            'total_count': self.count
        }
    
def labelwrite(start,high,low,close):
    if(start<=close and high<=close and start<=low):
        z=[0]    #low=start<end=high:1
    elif(start<=close and high>close and start<=low):
        z=[1]    #low=start<end<high:2
    elif(start<=close and high<=close and start>low):
        z=[2]     #low<start<end=high:3
    elif(start<=close and high>close and start>low):
        z=[3]      #low<start<end<high:4
    elif(start>close and high>start and close>low):
        z=[4]      #low=end<start=high:5
    elif(start>close and high<=start and close>low):
        z=[5]      #low=end<start<high:6
    elif(start>close and high>start and close<=low):
        z=[6]      #low<end<start=high:7
    elif(start>close and high<=start and close<=low):
        z=[7]      #low<end<start<high:8
    return z

def labelwrite_more(past_close, history_high, history_low, start, high, low, close):
    if(start<=close and high<=close and start<=low):
        z=0    #low=start<end=high:1
        min = start
        max = close
    elif(start<=close and high>close and start<=low):
        z=1    #low=start<end<high:2
        min = start
        max = close
    elif(start<=close and high<=close and start>low):
        z=2     #low<start<end=high:3
        min = start
        max = close
    elif(start<=close and high>close and start>low):
        z=3      #low<start<end<high:4
        min = start
        max = close
    elif(start>close and high>start and close>low):
        z=4      #low=end<start=high:5
        min = close
        max = start
    elif(start>close and high<=start and close>low):
        z=5      #low=end<start<high:6
        min = close
        max = start
    elif(start>close and high>start and close<=low):
        z=6      #low<end<start=high:7
        min = close
        max = start
    elif(start>close and high<=start and close<=low):
        z=7      #low<end<start<high:8
        min = close
        max = start
        
    if(past_close>max):
        m = 0
    elif(past_close>min):
        m = 1
    elif(past_close<=min):
        m = 2
        
    if(history_high==high and history_low==low):
        type = 0
    elif(history_high==high and history_low<low):
        type = 1
    elif(history_high>high and history_low==low):
        type = 2
    elif(history_high>high and history_low<low):
        type = 3
    
    return [(type*3+z)*3+m]

def labelwrite_bin(start,close):
    if(start<=close):
        z=[1,0]    #low=start<end=high:1
    else:
        z=[0,1]      #low<end<start<high:8
    return z
    
def DataProcess(file_num,data_path,train_path,test_path,train_percentage,num_features, window_size, initial_forget_factor, drift_threshold, min_forget_factor, max_forget_factor, normalise=False):
    z=['date','open','high','low','close',
        'open_month_volatility_21','open_month_volatility_32','open_month_volatility_65','open_month_volatility_96',
        'high_month_volatility_21','high_month_volatility_32','high_month_volatility_65','high_month_volatility_96',
        'low_month_volatility_21','low_month_volatility_32','low_month_volatility_65','low_month_volatility_96',
        'close_month_volatility_21','close_month_volatility_32','close_month_volatility_65','close_month_volatility_96',
        'open_year_volatility_21','open_year_volatility_32','open_year_volatility_65','open_year_volatility_96','open_year_volatility_127','open_year_volatility_190','open_year_volatility_252',
        'high_year_volatility_21','high_year_volatility_32','high_year_volatility_65','high_year_volatility_96','high_year_volatility_127','high_year_volatility_190','high_year_volatility_252',
        'low_year_volatility_21','low_year_volatility_32','low_year_volatility_65','low_year_volatility_96','low_year_volatility_127','low_year_volatility_190','low_year_volatility_252',
        'close_year_volatility_21','close_year_volatility_32','close_year_volatility_65','close_year_volatility_96','close_year_volatility_127','close_year_volatility_190','close_year_volatility_252',
        'open_ma_5','high_ma_5','low_ma_5','close_ma_5',
        'open_ma_10','high_ma_10','low_ma_10','close_ma_10',
        'open_ma_20','high_ma_20','low_ma_20','close_ma_20',
        'open_ma_30','high_ma_30','low_ma_30','close_ma_30',
        'open_ma_60','high_ma_60','low_ma_60','close_ma_60',
        'open_logma_5','high_logma_5','low_logma_5','close_logma_5',
        'open_logma_10','high_logma_10','low_logma_10','close_logma_10',
        'open_logma_20','high_logma_20','low_logma_20','close_logma_20',
        'open_logma_30','high_logma_30','low_logma_30','close_logma_30',
        'open_logma_60','high_logma_60','low_logma_60','close_logma_60',
        'open_short_EMA','high_short_EMA','low_short_EMA','close_short_EMA',
        'open_long_EMA','high_long_EMA','low_long_EMA','close_long_EMA',
        'open_DIF','high_DIF','low_DIF','close_DIF',
        'open_DEA','high_DEA','low_DEA','close_DEA',
        'open_MACD','high_MACD','low_MACD','close_MACD',
        'volume','amount','turn','preclose','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM',
        'K_value','D_value','J_value']
    name=['label_today', 'label_3day', 'label_week(5)', 'label_month(21)', 'label_3month(65)', 'label_6month(151)', 'label_year(252)']
    name_bin=['label_today0','label_today1',
            'label_3day0','label_3day1',
            'label_week(5)0','label_week(5)1',
            'label_month(21)0','label_month(21)1',
            'label_3month(65)0','label_3month(65)1',
            'label_6month(151)0','label_6month(151)1',
            'label_year(252)0','label_year(252)1',]
    for id in range(file_num):
        data = pd.read_csv(f"{data_path}/_{id}_better.csv",encoding="utf-8")
        data = np.array(data,dtype='float32')#将pandas读取的数据转化为array
        date=data[:,0]
        data=data[:,1:]   
        with open(f'{train_path}/data/_{id}.csv',"w+",encoding='utf-8',newline='') as train_data,open(f"{train_path}/label/_{id}.csv", "w+", encoding="utf-8", newline="") as train_label,open(f'{test_path}/data/_{id}.csv',"w+",encoding='utf-8',newline='') as test_data,open(f"{test_path}/label/_{id}.csv", "w+", encoding="utf-8", newline="") as test_label:
            train_data_writer = csv.writer(train_data)
            train_data_writer.writerow(z)
            train_label_writer = csv.writer(train_label)
            train_label_writer.writerow(name)
            #train_label_writer.writerow(name_bin)
            test_data_writer = csv.writer(test_data)
            test_data_writer.writerow(z)
            test_label_writer = csv.writer(test_label)
            test_label_writer.writerow(name)
            #test_label_writer.writerow(name_bin)
            
            # 数据维度
                # DIM = np.size(data,1)
            length=int(np.size(data,0)*train_percentage)
            
            # 创建一个归一化器实例
                # normalizer = IncrementalNormalizer(dim=DIM)                 
            norm = AdaptiveIncrementalNorm(num_features=num_features, window_size=window_size, initial_forget_factor=initial_forget_factor, drift_threshold=drift_threshold, min_forget_factor=min_forget_factor, max_forget_factor=max_forget_factor)
            # 增量处理数据
            idx=0
            # for x,normalized_x in zip(data,X_scaled):
            for x in data:
                x = torch.from_numpy(x)
                
                # 更新参数
                    # normalizer.update(x)
                normalized_x = norm.update_and_normalize(x)
                
                # 归一化当前数据点
                    # normalized_x = normalizer.normalize(x)
                    # normalized_x = torch.from_numpy(normalized_x)
                # normalized_x = F.normalize(normalized_x, p=2, dim=0)
                d=[date[idx].tolist()]
                if(normalise):
                    d.extend(normalized_x.tolist())
                else:
                    d.extend(x.tolist())
                
                dd=data[idx-1-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                today=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                
                # dd=data[idx-3-np.size(data,0):idx+1,3]
                # start=dd[0]
                # high=np.max(dd)
                # low=np.min(dd)
                # close=x[3]
                # _3day=labelwrite(start,high,low,close)
                dd=data[idx-4-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-3-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                _3day=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                
                # dd=data[idx-5-np.size(data,0):idx+1,3]
                # start=dd[0]
                # high=np.max(dd)
                # low=np.min(dd)
                # close=x[3]
                # week=labelwrite(start,high,low,close)
                dd=data[idx-6-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-5-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                week=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                #week=labelwrite_bin(start,close)
                
                # dd=data[idx-21-np.size(data,0):idx+1,3]
                # start=dd[0]
                # high=np.max(dd)
                # low=np.min(dd)
                # close=x[3]
                # month=labelwrite(start,high,low,close)
                dd=data[idx-22-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-21-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                month=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                #month=labelwrite_bin(start,close)
                
                # dd=data[idx-65-np.size(data,0):idx+1,3]
                # start=dd[0]
                # high=np.max(dd)
                # low=np.min(dd)
                # close=x[3]
                # _3month=labelwrite(start,high,low,close)
                dd=data[idx-66-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-65-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                _3month=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                #_3month=labelwrite_bin(start,close)
                
                # dd=data[idx-126-np.size(data,0):idx+1,3]
                # start=dd[0]
                # high=np.max(dd)
                # low=np.min(dd)
                # close=x[3]
                # _6month=labelwrite(start,high,low,close)
                dd=data[idx-127-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-126-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                _6month=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                #_6month=labelwrite_bin(start,close)
                
                # dd=data[idx-252-np.size(data,0):idx+1,3]
                # start=dd[0]
                # high=np.max(dd)
                # low=np.min(dd)
                # close=x[3]
                # year=labelwrite(start,high,low,close)
                dd=data[idx-253-np.size(data,0):idx+1,0:4]
                past_close = dd[0,3]
                dd=data[idx-252-np.size(data,0):idx+1,0:4]
                history_high = np.max(dd[:,1])
                history_low = np.min(dd[:,2])
                start=x[0]
                high=x[1]
                low=x[2]
                close=x[3]
                year=labelwrite_more(past_close, history_high, history_low, start,high,low,close)
                #year=labelwrite_bin(start,close)
                
                today.extend(_3day)
                today.extend(week)
                today.extend(month)
                today.extend(_3month)
                today.extend(_6month)
                today.extend(year)
                if(idx==0):
                    idx+=1
                    continue
                if(idx<=length):
                    train_data_writer.writerow(d)
                    train_label_writer.writerow(today)
                else:
                    test_data_writer.writerow(d)
                    test_label_writer.writerow(today)
                idx+=1
   
def LabelConstruct(data_path,istrain):
    for path in data_path:
        # 1. 创建文件对象（指定文件名，模式，编码方式）a模式 为 下次写入在这次的下一行
        with open(f'{path}.csv',"r+",encoding='utf-8',newline='') as d,open(f"{path}_label.csv", "w", encoding="utf-8", newline="") as f:
            csv_reader = csv.reader(d)
            csv_writer = csv.writer(f)
            name=['label0','label1','label2','label3','label4','label5','label6','label7'] if istrain else ['label']
            csv_writer.writerow(name)
            z=[0,0,0,0,0,0,0,0] if istrain else [-1]
            for row in csv_reader:        
                # 4. 写入csv文件内容
                #start2,high3,low4,end5
                if(row[1]=='open'):
                    continue
                row=[float(x) for x in row]
                if(row[3]==row[1] and row[4]==row[2] and row[1]<=row[4]):
                    z=[1,0,0,0,0,0,0,0] if istrain else [0]    #low=start<end=high:1
                elif(row[3]==row[1] and row[4]<=row[2] and row[1]<=row[4]):
                    z=[0,1,0,0,0,0,0,0] if istrain else [1]    #low=start<end<high:2
                elif(row[3]<=row[1] and row[4]==row[2] and row[1]<=row[4]):
                    z=[0,0,1,0,0,0,0,0] if istrain else [2]     #low<start<end=high:3
                elif(row[3]<=row[1] and row[4]<=row[2] and row[1]<=row[4]):
                    z=[0,0,0,1,0,0,0,0] if istrain else [3]      #low<start<end<high:4
                elif(row[3]==row[4] and row[1]==row[2] and row[1]>=row[4]):
                    z=[0,0,0,0,0,0,0,1] if istrain else [7]      #low=end<start=high:5
                elif(row[3]==row[4] and row[1]<=row[2] and row[1]>=row[4]):
                    z=[0,0,0,0,0,0,1,0] if istrain else [6]      #low=end<start<high:6
                elif(row[3]<=row[4] and row[1]==row[2] and row[1]>=row[4]):
                    z=[0,0,0,0,0,1,0,0] if istrain else [5]      #low<end<start=high:7
                elif(row[3]<=row[4] and row[1]<=row[2] and row[1]>=row[4]):
                    z=[0,0,0,0,1,0,0,0] if istrain else [4]      #low<end<start<high:8
                csv_writer.writerow(z)
            print("写入数据成功")
            # 5. 关闭文件
            f.close()
        
def BinLabelConstruct(data_path,istrain):
    for path in data_path:
        # 1. 创建文件对象（指定文件名，模式，编码方式）a模式 为 下次写入在这次的下一行
        with open(f'{path}.csv',"r+",encoding='utf-8',newline='') as d,open(f"{path}_label_bin.csv", "w", encoding="utf-8", newline="") as f:
            csv_reader = csv.reader(d)
            csv_writer = csv.writer(f)
            name=['label0','label1'] if istrain else ['label']
            csv_writer.writerow(name)
            z=[0,0] if istrain else [-1]
            for row in csv_reader:        
                # 4. 写入csv文件内容
                #start2,high3,low4,end5
                if(row[0]=='open'):
                    continue
                row=[float(x) for x in row]
                if(row[0]<=row[3]):
                    z=[1,0] if istrain else [0]    #low=start<end=high:1
                else:
                    z=[0,1] if istrain else [1]    #low=start<end<high:2
                
                csv_writer.writerow(z)
            print("写入数据成功")
            # 5. 关闭文件
            f.close()
    
if __name__ =='__main__':
    
    l=['600104','600206','600756','600839','600000','600109','600887','600690','600688','600198']
    l0=['F:/stockPre/data/better_day_data_test/_0_better',
        'F:/stockPre/data/better_day_data_test/_1_better',
        'F:/stockPre/data/better_day_data_test/_2_better',
        'F:/stockPre/data/better_day_data_test/_3_better',
        'F:/stockPre/data/better_day_data_test/_4_better',
        'F:/stockPre/data/better_day_data_test/_5_better',
        'F:/stockPre/data/better_day_data_test/_6_better',
        'F:/stockPre/data/better_day_data_test/_7_better',
        'F:/stockPre/data/better_day_data_test/_8_better',
        'F:/stockPre/data/better_day_data_test/_9_better',]
    l1=['F:/stockPre/data/better_day_data_test/_0',
        'F:/stockPre/data/better_day_data_test/_1',
        'F:/stockPre/data/better_day_data_test/_2',
        'F:/stockPre/data/better_day_data_test/_3',
        'F:/stockPre/data/better_day_data_test/_4',
        'F:/stockPre/data/better_day_data_test/_5',
        'F:/stockPre/data/better_day_data_test/_6',
        'F:/stockPre/data/better_day_data_test/_7',
        'F:/stockPre/data/better_day_data_test/_8',
        'F:/stockPre/data/better_day_data_test/_9',]
    l1=['F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_0',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_1',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_2',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_3',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_4',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_5',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_6',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_7',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_8',
        'F:/stockPre/data/data_ComplexFormed_complexLabel/resources/_9',]
    # DailyDataget(l,'F:/stockPre/data/data_ComplexFormed_complexLabel/resources','2000-01-01','2025-01-01')
    # betterCsv(l1)
    DataProcess(10,'F:/stockPre/data/data_ComplexFormed_complexLabel/resources','F:/stockPre/data/data_ComplexFormed_complexLabel/train','F:/stockPre/data/data_ComplexFormed_complexLabel/test',0.83, num_features=120, window_size=252, initial_forget_factor=0.8, drift_threshold=0.05, min_forget_factor=0.7, max_forget_factor=0.99, normalise=True)
    #LabelConstruct(l0,False)

    #LabelConstruct(l1,True)

# F:/stockPre/data/better_day_data/_0_better
# 0.5219100019650226 0.4780899980349774
# [0.024759284731774415, 0.05875417567302024, 0.013362153664767144, 0.4250343878954608, 0.39909608960503046, 0.06916879544114757, 0.0076635881312635094, 0.0021615248575358615]
# ==================================================
# F:/stockPre/data/better_day_data/_1_better
# 0.5576734132442523 0.4423265867557477
# [0.03360188642169385, 0.07447435645509923, 0.01906071919827078, 0.4305364511691884, 0.34446846138730597, 0.07683238357241108, 0.016309687561406955, 0.004716054234623698]
# ==================================================
# F:/stockPre/data/better_day_data/_2_better
# 0.562585969738652 0.437414030261348
# [0.04401650618982118, 0.06032619375122814, 0.030064845745726074, 0.4281784240518766, 0.3485950088426017, 0.06916879544114757, 0.015916683041854982, 0.003733542935743761]
# ==================================================
# F:/stockPre/data/better_day_data/_3_better
# 0.5482413047750049 0.45175869522499507
# [0.03261937512281391, 0.047553546865788957, 0.04519551974847711, 0.42287286303792493, 0.34643348398506585, 0.07467085871487522, 0.024562782471998428, 0.00609157005305561]
# ==================================================
# F:/stockPre/data/better_day_data/_4_better
# 0.5191589703281587 0.4808410296718412
# [0.024759284731774415, 0.050108076242876796, 0.015720180782078996, 0.42857142857142855, 0.3953625466692867, 0.06661426606405974, 0.014541167223423069, 0.004323049715071723]
# ==================================================
# F:/stockPre/data/better_day_data/_5_better
# 0.5454902731381411 0.4545097268618589
# [0.04106897229318137, 0.06857928866181962, 0.029475338966398115, 0.40636667321674197, 0.35566909019453724, 0.07545686775397917, 0.018078207899390842, 0.0053055610139516604]
# ==================================================
# F:/stockPre/data/better_day_data/_6_better
# 0.5350756533700137 0.4649243466299862
# [0.024562782471998428, 0.07074081351935547, 0.011200628807231282, 0.42857142857142855, 0.37551581843191195, 0.07781489487129102, 0.008646099430143446, 0.0029475338966398115]
# ==================================================
# F:/stockPre/data/better_day_data/_7_better
# 0.5331106307722538 0.4668893692277461
# [0.034584397720573784, 0.06740027510316368, 0.017881705639614855, 0.41324425230890155, 0.3818038907447436, 0.07290233837689133, 0.01002161524857536, 0.0021615248575358615]
# ==================================================
# F:/stockPre/data/better_day_data/_8_better
# 0.5600314403615642 0.43996855963843584
# [0.03144036156415799, 0.05836117115346826, 0.0485360581646689, 0.421693849479269, 0.3199056789153075, 0.08410296718412262, 0.026134800550206328, 0.009825112988799371]
# ==================================================
# F:/stockPre/data/better_day_data/_9_better
# 0.5616034584397721 0.43839654156022795
# [0.06641776380428376, 0.056985655335036356, 0.03144036156415799, 0.40675967773629396, 0.3505600314403616, 0.06032619375122814, 0.021025741796030655, 0.006484574572607585]

#text数据集类别比例：
#0.5139318885448917 0.48606811145510836
# 0.0041279669762641896  0
# 0.05779153766769866    1
# 0.04231166150670795    2
# 0.40970072239422084    3
# 0.3364293085655315     4
# 0.07017543859649122    5
# 0.06914344685242518    6
# 0.010319917440660475   7

#train数据集类别比例：
# 0.5191589703281587 0.4808410296718412
# 0.024759284731774415
# 0.050108076242876796
# 0.015720180782078996
# 0.42857142857142855
# 0.3953625466692867
# 0.06661426606405974
# 0.014541167223423069
# 0.004323049715071723