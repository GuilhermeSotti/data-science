import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statistics
from matplotlib import style
from sklearn import svm, preprocessing
from collections import Counter 

style.use("dark_background")

def Forward(features = ['DE Ratio',
                        'Trailing P/E',
                        'Price/Sales',
                        'Price/Book',
                        'Profit Margin',
                        'Operating Margin',
                        'Return on Assets',
                        'Return on Equity',
                        'Revenue Per Share',
                        'Market Cap',
                        'Enterprise Value',
                        'Forward P/E',
                        'PEG Ratio',
                        'Enterprise Value/Revenue',
                        'Enterprise Value/EBITDA',
                        'Revenue',
                        'Gross Profit',
                        'EBITDA',
                        'Net Income Avl to Common ',
                        'Diluted EPS',
                        'Earnings Growth',
                        'Revenue Growth',
                        'Total Cash',
                        'Total Cash Per Share',
                        'Total Debt',
                        'Current Ratio',
                        'Book Value Per Share',
                        'Cash Flow',
                        'Beta',
                        'Held by Insiders',
                        'Held by Institutions',
                        'Shares Short (as of',
                        'Short Ratio',
                        'Short % of Float',
                        'Shares Short (prior ']

  data_df = pd.DataFrame.from_csv("key_stats_acc_perf_NO_NA.csv")
  #data_df = data_df[:, 100]
  data_df = data_df.reindex(np.random.permutation(data_df.index))
  data_df = data_df.replace("NaN", 0).replace("N/A", 0)
  X = np.array(data_df[features].values.tolist())
  Y = (data_df["Status"]
       .replace("underperform", 0)
       .replace("outperform", 1)
       .values.tolist())
  Z = np.array(data_df[["stock_p_change", "sp500_p_change"]]
               
  return X, Y, Z

def Analysis():

  test_size = 500
  invest_amount = 10000
  total_invests = 0
  if_market = 0
  if_strat = 0
  X, Y, Z = Build_Data_Set()
  print(len(X))
  clf = svm.SVC(kernel = "linear", C = 1.0)
  clf.fit(X, Y)
  
  correct_cont = 0
  for x in range(1, test_size + 1):
    if clf.predict(X[-x])[0] == y[-x]:
      correct_cont =+ 1

    if clf.predict(X[-x])[0] == 1:
      invest_return = invest_amount + (invest_amount * (Z[-x][0]/100))
      market_return = invest_amount + (invest_amount * (Z[-x][1]/100))
      total_invests += 1
      if_market += 1
      if_strat += market_return


  do_nothing = total_invests * invests_amount

  # print("Accuracy: ", (correct_cont/ test_size) * 100.00)
  # print("Total trades: ", total_invests)
  # print("Ending with strategy: ", if_strat)
  # print("Ending with market: ", if_market)
  # print("Compared to market, we earn ", str(((if_strat - if_market) / if_market) * 100.00 ), " % more")
  # print("Compared to market, we earn ", str(((if_market - do_nothing) / do_nothing)) * 100.00), " % more")
  # print("Compared to market, we earn ", str(((if_strat - do_nothing) / do_nothing) * 100.00), " % more")

  data_df = pd.DataFrame.from_csv("foward_sapmle_NO_N/A.csv")
  data_df = pd.replace("N/A", 0).replace("NaN", 0)
  X = np.array(data_df[features].values
  X = preprocessing.scale(X)
  Z = data_df["Ticker"].values.tolist()

  invest_list = []
  for i in range(len(x)):
    predict = clt.predict(X[i]) [0]
    if predict == 1:
      print(Z[i])
      invest_list.append(Z[i])

  print(len(invest_list))
  print(invest_list)    
  return invest_list

  #w = clf.coef_[0]
  #a = -w[0] / w[1]
  #xx = np.linspace(min(X[:, 0]), max(X[:, 0 ]))
  #yy = a * xx - clf.intercept_[0] / w[1]
  #h0 = plt.plot(xx, yy, "k-", 
                #label = "non weigthed",
                #c = y) 
  
  plt.legend()
  plt.show()

final_list = []
loops = 5
for x in range(loops)
  stock_list = Analysis()
  for e in stock_list:
    final_list.append(e)


x = Counter(final_list)
print(15 * '__')
for each in x:
  if x[each] > loops - (loops / 3)
    print(each)


def Randomizer()

  df = pd.DataFrame({"D1" range(5), "D2": range(5)})
  print(df)
  df2 = df.reindex(np.random.permutation(np.index))
  print(df2)
