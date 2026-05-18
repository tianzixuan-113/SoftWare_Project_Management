import baostock as bs
import pandas as pd
import numpy as np

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

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

for x in range(2007,2025):
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
        rs_profit = bs.query_profit_data(code="sh.600206", year=x, quarter=y)
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

filled_df.to_csv('F:/stockPre/data_get/merged_data_with_zero_filling.csv', index=False)

# 登出系统
bs.logout()
l=['600104','600206','600756','600839','600000','600109','600887','600690','600688','600198']
