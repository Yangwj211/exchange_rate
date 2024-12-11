import yfinance as yf
import pandas as pd

# 获取人民币对新加坡元（CNYSGD=X）的汇率数据
ticker = 'CNYSGD=X'

# 设置时间范围（近十年）
start_date = '2013-01-01'
end_date = '2023-01-01'

# 使用 yfinance 获取数据
data = yf.download(ticker, start=start_date, end=end_date)

# 打印数据的前几行，查看数据结构
print(data.head())

# 保存数据为 CSV 文件
data.to_csv('CNYSGD_2013_2023.csv')

# 提示用户已保存文件
print("数据已保存为 CNYSGD_2013_2023.csv")
