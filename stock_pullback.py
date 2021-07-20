#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Variables: n days to hold (probably 1, but as many as 10 [2 wks]),
# N top considered,
# max N top chosen,
# length of LONG period,
# length of SHORT period
# Backtest a subset of time
# Get the top N performers over the last LONG period
# Get the bottom N performers over the last SHORT period
# Find the intersection between those two
# Return THAT as a list
# Which ends up getting saved as a new column,
# Because it was done across the entire dataset's length


# ### CONSIDERATIONS
#
# Should functions be defined by their start row or their end row?
# Typically, we want to know something about this row based on historical data,
# which is to say based on PREVIOUS rows
#
# If a function is defined by its start row, then we are incorrectly learning based on the future.
#
# Further investigation is needed into holding positions for 2 days and trading every other day.

# In[223]:


import pandas as pd
import numpy as np
from pandas_datareader import data
# from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
from sklearn.model_selection import train_test_split

import os
import smtplib
from email.mime import multipart, text
from email import mime


# In[167]:


### Functions ###
##-------------##

def calculate_hold_score(dataset, time_period, fund, end):
    suffix = "_Adj Close"
    ### This is me holding an index fund for time_period days
    if not fund.endswith(suffix):
        fund = "".join([fund, suffix])

#     end = start + time_period
    start = end - time_period
    return (constant[fund].iloc[end] -             constant[fund].iloc[start]) / constant[fund].iloc[start]

def make_pullbacks(dataset, long, short, top_n, end, symbols, force_buy=False):
    suffix = "_Adj Close"
    four_mo_performance_dict = {}
#     end = start + long
    start = end - long
    start -= 1
    end -= 1
    for col in symbols:
        if not col.endswith(suffix):
            col = "".join([col, suffix])
        four_mo_performance_dict[col] =                 (dataset[col][end] - dataset[col][start]) /                 dataset[col][start]

    best_appreciators = []
    for _ in range(top_n):
        the_max = max(four_mo_performance_dict.values())
        best_appreciators.extend([key for key in four_mo_performance_dict.keys()                 if the_max == four_mo_performance_dict[key]])
        del four_mo_performance_dict[best_appreciators[-1]]


    # Same thing, but over short days, and getting the min
    # end_date = -1
    ten_day_performance_dict = {}
    end = start + short
    for col in symbols:
        if not col.endswith(suffix):
            col = "".join([col, suffix])
        ten_day_performance_dict[col] =                 (dataset[col][end] - dataset[col][start]) /                 dataset[col][start]

    worst_appreciators = []
    for n in range(top_n):
        the_min = min(ten_day_performance_dict.values())
        worst_appreciators.extend([key for key in ten_day_performance_dict.keys()                 if the_min == ten_day_performance_dict[key]])
        del ten_day_performance_dict[worst_appreciators[-1]]

    intersection = set(best_appreciators) & set(worst_appreciators)

    ### INSTEAD OF THESE NEXT 2 LINES,
    ### you may ffill empty lists in the produced column
    if force_buy and len(intersection) < 1:
        return make_pullbacks(dataset, long, short, top_n+1, end)

    intersection = list(intersection)
    intersection.sort(key=lambda x: best_appreciators.index(x))
#     print(best_appreciators, worst_appreciators)
    return list(intersection)

def make_tops(dataset, long, top_n, end, symbols):
    suffix = "_Adj Close"
    four_mo_performance_dict = {}
#     end = start + long
    start = end - long
    start -= 1
    end -= 1
    for col in symbols:
        if not col.endswith(suffix):
            col = "".join([col, suffix])
        four_mo_performance_dict[col] =                 (dataset[col][end] -                 dataset[col][start]) /                 dataset[col][start]

    best_appreciators = []
    for _ in range(top_n):
        the_max = max(four_mo_performance_dict.values())
        best_appreciators.extend([key for key in four_mo_performance_dict.keys()                 if the_max == four_mo_performance_dict[key]])
        del four_mo_performance_dict[best_appreciators[-1]]

    return list(best_appreciators)

def as_column(built_list, dataframe, shift=False):
    if shift:
        new_col = pd.DataFrame([np.nan] * (len(dataframe) - len(built_list) - 1))
        new_col = new_col.append(built_list)
        new_col = new_col.append([np.nan])
    else:
        new_col = pd.DataFrame([np.nan] * (len(dataframe) - len(built_list)))
        new_col = new_col.append(built_list)
    new_col.index = dataframe.index
    return new_col

def make_buy_hold_pullbacks(pullback_col, top_col, long, dataframe):
    new_col = []
    for j, i in enumerate(range(long, len(dataframe))):
        new_col.append(dataframe.iloc[i][pullback_col])
        if j != 0:
            for value in new_col[j-1]:
                if value not in new_col[j]:
                    new_col[j].append(value)
        new_col[j] = list(set(new_col[j]) & set(dataframe.iloc[i][top_col]))
        new_col[j].sort(key=lambda item: dataframe.iloc[i][top_col].index(item))

    return tuple(new_col)

def write_to_file(message, filepath):
    with open(filepath, 'a') as outfile:
        outfile.write(str(message))
        outfile.write("\n")

# In[4]:


# Variables:
## n days to hold (probably 1, but as many as 10 [2 wks]), // try 1 and 2
## N top considered,            // try 13 and 15
## max N top chosen,            // try 13 and 15
## length of LONG period,      // try 75 and 140
## length of SHORT period      // try 7 and 10
# Backtest a subset of time
# Get the top N performers over the last LONG period
# Get the bottom N performers over the last SHORT period
# Find the intersection between those two
# Return THAT as a list
# Which ends up getting saved as a new column,
# Because it was done across the entire dataset's length

# path = "C:\\Users\\Thalo\\Development\\project_backtest\\"
path = "/home/thalo/Development/"

filename = "stock_output.log"
suffix = "_Adj Close"
# vanguard_funds = ['BIV', 'BLV', 'BND', 'BNDW', 'BNDX', 'BSV', 'EDV', 'ESGV', 'IVOG', 'IVOO', 'IVOV', 'MGC', 'MGK', 'MGV', 'VAW', 'VB', 'VBK', 'VBR', 'VCIT', 'VCLT', 'VCR', 'VCSH', 'VDC', 'VDE', 'VEA', 'VEU', 'VFH', 'VFLQ', 'VFMF', 'VFMO', 'VFMV', 'VFQY', 'VFVA', 'VGIT', 'VGK', 'VGLT', 'VGSH', 'VGT', 'VHT', 'VIG', 'VIGI', 'VIOG', 'VIOO', 'VIOV', 'VIS', 'VMBS', 'VNQ', 'VNQI', 'VO', 'VOE', 'VONE', 'VONG', 'VONV', 'VOO', 'VOOG', 'VOOV', 'VOT', 'VOX', 'VPL', 'VPU', 'VSGX', 'VSS', 'VT', 'VTC', 'VTEB', 'VTHR', 'VTI', 'VTIP', 'VTV', 'VTWG', 'VTWO', 'VTWV', 'VUG', 'VV', 'VWO', 'VWOB', 'VXF', 'VXUS', 'VYM', 'VYMI']
good_funds = ['VGT', ]# 'VCR', 'VPU', 'EDV', 'VDE']

# In[155]:


### Get data ###
##------------##

new_dict = {}
symbols = [
    'LIN', 'ECL', 'IFF', 'FMC', 'CF', 'ALB', 'AXTA', 'NKE', 'CL', 'PM', 'EL', 'ATVI', 
    'EA', 'TSLA', 'STZ', 'MNST', 'LULU', 'HSY', 'MKC', 'CHD', 'BF-B', 'TTWO', 'NVR', 
    'CLX', 'LW', 'W', 'FBHS', 'LKQ', 'DHI', 'LEN', 'WBC', 'HRL', 'UAA', 'UA', 
    'LEN-B', 'AMZN', 'HD', 'CMCSA', 'NFLX', 'MCD', 'DIS', 'COST', 'SBUX', 'BKNG', 'LOW', 
    'CHTR', 'TJX', 'MAR', 'ROST', 'DG', 'YUM', 'ORLY', 'HLT', 'AZO', 'DLTR', 'CMG', 
    'ULTA', 'EXPE', 'CPRT', 'TSCO', 'LUV', 'DPZ', 'WYNN', 'FDS', 'RCL', 'LYV', 'ATUS', 
    'UBER', 'SIRI', 'MGM', 'ROL', 'TIF', 'MTN', 'LYFT', 'TRIP', 'CHWY', 'CVNA', 
    'H', 'V', 'MA', 'AMT', 'SPGI', 'CCI', 'MMC', 'PLD', 'ICE', 'SCHW', 'SPG', 'AON', 
    'EQIX', 'PSA', 'MCO', 'AVB', 'SBAC', 'TROW', 'DLR', 'INFO', 'O', 'BXP', 'MSCI', 
    'ESS', 'WELL', 'CBRE', 'EFX', 'FRC', 'AJG', 'ARE', 'MKL', 'AMTD', 'MAA', 'EXR', 
    'UDR', 'SIVB', 'CBOE', 'REG', 'INVH', 'VNO', 'ETFC', 'FRT', 'IRM', 'SEIC', 'RJF', 
    'CPT', 'IBKR', 'TMO', 'BMY', 'BDX',  'SYK', 'ISRG', 'BSX', 'ZTS', 'ILMN', 
    'ABBV', 'VRTX', 'EW', 'ALXN', 'IQV', 'REGN', 'IDXX', 'BIIB', 'CNC', 'BAX',
    'ALGN', 'RMD', 'COO', 'INCY', 'BMRN', 'TFX', 'DXCM', 'VAR', 'ABMD', 'ALNY', 
    'JAZZ', 'EXAS', 'SGEN', 'NKTR', 'BA', 'PYPL', 'UNP', 'ACN', 'LMT', 'UPS', 'ADP', 
    'MMM', 'DHR', 'FIS', 'SHW', 'ROP', 'FISV', 'APH', 'PAYX', 'WCN', 'GPN', 
    'SQ', 'FLT', 'A', 'FTV', 'TDG', 'VRSK', 'ITW', 'CTAS', 'MTD', 'AME', 'CSGP', 
    'ROK', 'FAST', 'VMC', 'KEYS', 'XYL', 'BR', 'MLM', 'WAT', 'TRU', 'EXPD', 'KSU', 
    'MAS', 'WAB', 'TRMB', 'ODFL', 'JBHT', 'RHI', 'CHRW', 'IPGP', 'URI', 'JKHY', 
    'PKG', 'ST', 'CGNX', 'HEI-A', 'FLIR', 'HUBB', 'AOS', 'HEI', 'XPO', 'EOG',  
    'OKE', 'PXD', 'CXO', 'OXY', 'FANG', 'LNG', 'APA', 'NBL', 'COG', 'TRGP', 'CLR', 
    'XEC', 'MSFT', 'AAPL', 'FB', 'GOOGL', 'GOOG', 'ADBE', 'CRM', 'AVGO', 'TXN', 
    'NVDA', 'INTU', 'NOW', 'MU', 'AMAT', 'ADI', 'ADSK', 'WDAY',  'XLNX', 'AMD',
    'LRCX', 'TWTR', 'CERN', 'VRSN', 'LHX', 'VEEV', 'MCHP', 'CDNS', 'SNPS', 'KLAC', 
    'SPLK', 'PANW', 'CTSH', 'ANSS', 'MTCH', 'MXIM', 'TWLO', 'ANET', 'IT', 'MSI', 'SWKS', 
    'VMW', 'AKAM', 'SSNC', 'GDDY', 'CTXS', 'FTNT', 'SNAP', 'FFIV', 'BKI', 'OKTA', 
    'PAYC', 'DBX', 'WORK', 'CDK', 'PINS', 'CRWD', 'TMUS', 'ZAYO', 'ZM', 'NRG'
    # 'RHT', 'WP', 'APC', 'TSS', 'WCG', 'CELG',
        ]
symbols.extend(good_funds)
symbols = list(set(symbols))

# if os.path.exists("C:\\Users\\Thalo\\Development\\project_backtest\\stocks{}.csv".format(np.datetime64('today'))):
#     constant = pd.read_csv("C:\\Users\\Thalo\\Development\\project_backtest\\stocks{}.csv".format(np.datetime64('today')))
#     constant.set_index("Date", inplace=True)
# else:
# Reading from Stooq
error_symbols = []
for x in symbols:
    try:
        # y = data.DataReader(x, 'stooq')
        y = data.DataReader(x, 'yahoo', np.datetime64('today') - np.timedelta64(52, 'W'), np.datetime64('today'))
        if len(y) < 1:
            raise Exception("Got nothing back")
        new_dict[x] = y
    except:
        print("Error with:", x)
        error_symbols.append(x)
for x in error_symbols:
    symbols.remove(x)

# Preprocessing
for ticker in new_dict.keys():
    new_dict[ticker].columns = list(map(lambda x: ticker + "_" + x, new_dict[ticker].columns))

# Merging everything together
constant = new_dict[symbols[0]]
the_rest = list(new_dict.keys())[1:]
for ticker in the_rest:
    # print(ticker)

    constant = pd.merge(constant, new_dict[ticker],
              left_index=True, right_index=True,
              how='outer')

del new_dict
# constant.to_csv("C:\\Users\\Thalo\\Development\\project_backtest\\stocks{}.csv".format(np.datetime64('today')))

# In[156]:


# good_funds = []
# max_funds = {}
# # The length of time to hold a position
# time_period = 7
# print(delta, delta+time_period)
# funds = ['VTI', 'VTV', 'VUG', 'VGT', 'MGK', 'VEA', 'VFH', \
#         'VHT', 'VOO', 'VDC', 'VCR', 'VOX', 'VOOG', 'VBK', \
#         'VDE', 'VPU', 'VB']
"""
for _ in range(5):
    i = 0
    while i < (len(constant)-time_period) // 10:
        delta = np.random.randint(1, len(constant))
        fund_scores = {}
        for fund in vanguard_funds:
    #         print(fund, end=" ")
            f = calculate_hold_score(constant, time_period, fund, delta)
            fund_scores[fund] = f
    #         print(round(f, 4))
        m = max(fund_scores.keys(), key=lambda x : fund_scores[x])
    #     print(m)
        if m in max_funds.keys():
            max_funds[m] += 1
        else:
            max_funds[m] = 1
        i += 1
    # print(max_funds)
    for _ in range(4):
        n = max(max_funds.keys(), key=lambda x : max_funds[x])
        good_funds.append(n)
        # print(n, end=" ")
        del max_funds[n]
    print()
good_funds = list(set(good_funds))
# print("Good funds:", good_funds)
"""

# In[ ]:


# above_line = [max([calculate_hold_score(constant, 1, fund, i) for fund in good_funds]) for i in range(75, len(constant))]
# constant['best_index_holds'] = as_column(above_line, constant)


# In[ ]:





# In[377]:


# Variables

SHORT = 7
# SHORT = 10

LONG = 75
# LONG = 140

TOP_N = 14
# TOP_N = 15

## Leave this on 13
MAX_N = 15
# MAX_N = 15

PERIOD = 1
# PERIOD = 2

AVG = 'unweighted'

# [x] 10, 140, 13, 13, 1, 'unweighted'
# [ ] 10, 140, 13, 13, 1, 'weighted'
# [x] 10, 140, 13, 13, 2, 'unweighted'
# [x] 10, 140, 13, 15, 1, 'unweighted'
# [x] 10, 140, 15, 13, 1, 'unweighted'
# [x] 10,  75, 13, 13, 1, 'unweighted'
# [x] 10,  75, 15, 13, 1, 'unweighted'
# [x]  7, 140, 13, 13, 1, 'unweighted'
# [ ]  7, 140, 13, 13, 1, 'weighted'
# [ ]  7, 140, 13, 13, 2, 'unweighted'
# [x]  7, 140, 13, 15, 1, 'unweighted'
# [x]  7, 140, 15, 13, 1, 'unweighted'
# [x]  7,  75, 13, 13, 1, 'unweighted'
# [x]  7,  75, 13, 13, 2, 'unweighted' *
# [x]  7,  75, 15, 13, 1, 'unweighted' *
# [x]  7,  75, 15, 13, 2, 'unweighted' *

# WINNER_SO_FAR = {SHORT: 7, LONG: 75, TOP_N: 15, MAX_N: 13, PERIOD: 1, AVG: 'unweighted'}



# In[378]:



for fund in good_funds:
    vang_scores = [calculate_hold_score(constant, PERIOD, fund, i) for i in range(1, len(constant))]
    constant[fund + '_1D_return'] = as_column(vang_scores, constant)



# In[379]:


# def calculate_hold_score(dataset, time_period, fund, start)
# def make_pullbacks(dataset, long, short, top_n, start, symbols)
# def make_tops(dataset, long, top_n, start, symbols)
# def as_column(built_list, dataframe)
# def make_buy_hold_pullbacks(pullback_col, top_col, long, dataframe)

# LONG = 140
# SHORT = 10
# TOP_N = 13
force_buy = False
pullback_name_col = [(make_pullbacks(constant, long=LONG, short=SHORT, top_n=TOP_N, end=i, symbols=symbols), )
                     for i in range(LONG, len(constant))]
constant['pullback_names'] = as_column(pullback_name_col, constant, shift=False)

if not force_buy:
    ### Are ALL 4 of these lines necessary?
    ### Not that it's a big deal...
    constant['pullback_names'] = constant['pullback_names'].replace('[]', np.nan)
    constant['pullback_names'].fillna(method="ffill", inplace=True)
    constant['pullback_names'][constant['pullback_names'].str.len() == 0] = np.nan
    constant['pullback_names'].fillna(method="ffill", inplace=True)


# In[380]:


### Is this score really necessary? Like this is an incomplete column to be sure

# PERIOD = 1
pullback_score_col = [np.array([calculate_hold_score(constant, PERIOD, x, i)
         for x in constant['pullback_names'].iloc[i]]).mean()
         for i in range(LONG, len(constant)) if type (constant['pullback_names'].iloc[i]) != float]
constant['pullback_score'] = as_column(pullback_score_col, constant)


# In[381]:


# LONG = 140
# MAX_N = 13
top_name_col = [(make_tops(constant, LONG, MAX_N, i, symbols),) for i in range(LONG, len(constant)+1)]
constant['top_13_names'] = as_column(top_name_col, constant, shift=False)


# In[382]:


top_score_col = [np.array([calculate_hold_score(constant, PERIOD, x, i)
         for x in constant['top_13_names'].iloc[i]]).mean()
         for i in range(LONG, len(constant)) if type(constant['top_13_names'].iloc[i]) != float]

constant['top_13_score'] = as_column(top_score_col, constant)


# In[383]:

if constant.pullback_names.iloc[LONG:LONG*2+1].isna().sum() > constant.top_13_names.iloc[LONG:LONG*2+1].isna().sum():
    num_shift = constant.pullback_names.iloc[LONG:LONG*2+1].isna().sum()
else:
    num_shift = constant.top_13_names.iloc[LONG:LONG*2+1].isna().sum()
buy_hold_pullback_col = make_buy_hold_pullbacks("pullback_names", "top_13_names", LONG+num_shift, constant)

# Hacky solution, but...
constant['buy_hold_pullback_names'] = as_column(pd.DataFrame([buy_hold_pullback_col, [np.nan] * len(buy_hold_pullback_col)]).T, constant)[0]


# In[384]:


buy_hold_pullback_score_col = [np.array([calculate_hold_score(constant, PERIOD, x, i)
         for x in constant['buy_hold_pullback_names'].iloc[i]]).mean()
         for i in range(LONG, len(constant)) if type (constant['buy_hold_pullback_names'].iloc[i]) != float]

constant['buy_hold_pb_score'] = as_column(buy_hold_pullback_score_col, constant)


# In[385]:


alt_message = constant.iloc[140:, -(7+len(good_funds)):]


# In[399]:


# total = 6870
with open('/home/thalo/Development/config/in_total.txt') as total_file:
    total = total_file.read().strip()
    total = int(float(total))


message = (constant.iloc[-1]['pullback_names'],)
sub_message=[round(total/len(constant.iloc[-1]['pullback_names'])/constant.iloc[-1][name],4) for name in constant.iloc[-1]['pullback_names']]
message_tuples=list(zip(message[0],sub_message))
message=""
for pair in message_tuples:
    message="\n".join([message,str(pair)])
message+=" ".join(["\n", str(len(constant.iloc[-1]['pullback_names'])), str(round(total/len(constant.iloc[-1]['pullback_names']), 4))])
write_to_file(str(message), path + filename)

message2 = (constant.iloc[-1]['top_13_names'],)
sub_message = [round(total / len(constant.iloc[-1]['top_13_names']) / constant.iloc[-1][name], 4) for name in constant.iloc[-1]['top_13_names']]
message_tuples = list(zip(message2[0], sub_message))
message2 = ""
for pair in message_tuples:
    message2 = '\n'.join([message2, str(pair)])
message2 += " ".join(['\n', str(len(constant.iloc[-1]['top_13_names'])), str(round(total/len(constant.iloc[-1]['top_13_names']), 4))])
write_to_file(str(message2), path + filename)


message3 = (constant.iloc[-1]['buy_hold_pullback_names'],)
sub_message=[round(total/len(constant.iloc[-1]['buy_hold_pullback_names'])/constant.iloc[-1][name],4) for name in constant.iloc[-1]['buy_hold_pullback_names']]
message_tuples=list(zip(message3[0],sub_message))
message3=""
for pair in message_tuples:
    message3="\n".join([message3,str(pair)])
message3 += " ".join(["\n", str(len(constant.iloc[-1]['buy_hold_pullback_names'])), str(round(total/len(constant.iloc[-1]['buy_hold_pullback_names']),4))])
write_to_file(str(message3), path + filename)

post_email = True
if post_email:
    # with open("C:\\Users\\Thalo\\Development\\project_backtest\\console.txt", "a") as logfile:
    with open('/home/thalo/Development/stock_console.log', 'a') as logfile:
        try:
            s = smtplib.SMTP(host="mail.privateemail.com", port=587)
            s.starttls()

            # with open("C:\\Users\\Thalo\\Documents\\admin_ref.txt") as f:
            with open("/home/thalo/Documents/admin_references.txt") as f:
                op = f.readlines()
                first, second = op
                first = first.strip()
                second = second.strip()
            s.login(first, second)

            msg = multipart.MIMEMultipart()
            msg["From"]="admin@thalo.xyz"
            msg["To"]="thalo_m@yahoo.com"
            msg["Subject"]="Today's Message Is {}".format(np.datetime64('today'))
            msg.attach(text.MIMEText(str(message), 'plain'))

            msg.attach(text.MIMEText("".join(["\n", str(message2)]), 'plain'))
            msg.attach(text.MIMEText("".join(["\n", str(message3)]), 'plain'))
            # msg.attach(text.MIMEText(str(d), 'plain'))
            # msg.attach(text.MIMEText(str(df['strats_' + str(d) + 'd_cum_return'].iloc[-1]), 'plain'))

            # msg.attach(text.MIMEText(s_build))
            s.send_message(msg)
        except Exception as e:
            logfile.write(str(e))
            logfile.write(str(np.datetime64('today')))

# 2 files are:
#   Development/stock_console.log to log errors
#   Development/stock_output.log to log output messages
