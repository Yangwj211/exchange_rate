import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas_datareader.data as web
import datetime

# 1. 获取近10年人民币兑新加坡元的汇率数据
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2023, 1, 1)

# 获取汇率数据（CNY to SGD）
data = web.DataReader("CNYSGD=X", "yahoo", start, end)

# 打印数据的前几行，查看数据结构
print(data.head())

# 2. 数据预处理
# 假设我们以每月的最后一个汇率为准
data['Date'] = data.index
data['Month'] = data['Date'].dt.to_period('M')

# 保留每月的最后一个汇率
monthly_data = data.groupby('Month').last()

# 选择收盘价（汇率）
monthly_data = monthly_data[['Close']].reset_index()
monthly_data.columns = ['Date', 'ExchangeRate']

# 绘制汇率图表
plt.plot(monthly_data['Date'], monthly_data['ExchangeRate'])
plt.xlabel('Date')
plt.ylabel('CNY to SGD')
plt.title('CNY to SGD Exchange Rate (2013-2023)')
plt.show()

# 3. 特征和目标变量的构建
# 目标是根据历史数据预测汇率的变化趋势，从而确定最佳的参数 a、b、x、y

# 计算汇率的日变化（简单差分）
monthly_data['RateChange'] = monthly_data['ExchangeRate'].diff()

# 4. 模拟策略并评估
# 为了简单起见，我们可以先用网格搜索来寻找一个简单模型的参数（a、b）作为示例

# 构建一个简单的回归模型来预测汇率变化
X = monthly_data[['ExchangeRate']]  # 输入特征
y = monthly_data['RateChange']  # 目标：汇率变化

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 使用随机森林回归模型训练数据
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 预测并计算误差
y_pred = model.predict(X_test)
error = np.mean(np.abs(y_test - y_pred))

print(f'Model Mean Absolute Error: {error}')

# 5. 网格搜索优化参数（a、b）
# 网格搜索可以帮助我们找到一个简单的策略中合适的阈值 a 和 b
# 这里我们假设 a = 0.190, b = 0.195 作为策略的参数

# 我们将模拟一个简单的策略：根据预测的汇率变化，选择合适的兑换时机
def simulate_strategy(data, model, a=0.190, b=0.195, x=100000, y=50000):
    capital = 1000000  # 初始资金
    capital_history = [capital]
    
    for i in range(1, len(data)):
        rate_change = model.predict(np.array([[data.iloc[i-1]['ExchangeRate']]]))[0]
        
        if data.iloc[i]['ExchangeRate'] < a:
            # 汇率低于a时兑换 x 元
            capital -= x
        elif data.iloc[i]['ExchangeRate'] > b:
            # 汇率高于b时兑换 y 元
            capital -= y
        
        # 记录每次操作后的资金
        capital_history.append(capital)
    
    return capital_history

# 使用模型预测并模拟策略
capital_history = simulate_strategy(monthly_data, model, a=0.190, b=0.195)

# 绘制模拟策略的资金变化曲线
plt.plot(monthly_data['Date'], capital_history)
plt.xlabel('Date')
plt.ylabel('Capital (SGD)')
plt.title('Strategy Capital Over Time')
plt.show()

# 6. 总结：优化模型参数
# 在实际应用中，优化参数（a, b, x, y）可以通过更多的特征、回测策略等手段来完成
