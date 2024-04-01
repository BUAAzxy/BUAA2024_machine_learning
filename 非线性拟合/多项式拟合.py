import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore")

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 准备数据
X_train = train_data[['x']]
y_train = train_data['y']
X_test = test_data[['x']]
y_test = test_data['y']

# 存储损失值，以便最后绘制趋势图
loss_list=[]

# 选择最佳的多项式回归模型
for degree_best in range(1,101):  
    poly_model_best = make_pipeline(PolynomialFeatures(degree_best), LinearRegression())
    poly_model_best.fit(X_train, y_train)
    y_pred_best = poly_model_best.predict(X_test)

    # 计算 MAE
    loss_best = mean_absolute_error(y_test, y_pred_best)
    print(f'Best Model (Degree {degree_best}) MAE:', loss_best)
    loss_list.append(loss_best)
    # 创建用于绘图的预测范围
    x_range = np.linspace(X_train['x'].min(), X_train['x'].max(), 500)
    y_range_pred = poly_model_best.predict(x_range.reshape(-1, 1))

    if degree_best in [5,8,10,15,20,35,50,65]:
        # 训练集绘图
        plt.figure(figsize=[10, 6])
        plt.scatter(X_train['x'], y_train, color='blue', alpha=0.7, label='Training Data')  # 训练数据
        plt.plot(x_range, y_range_pred, color='green', label=f'Polynomial Degree {degree_best} Fit')  # 模型预测
        plt.title('Polynomial Regression Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 测试集绘图
        plt.figure(figsize=[10, 6])
        plt.scatter(X_test['x'], y_test, color='red', alpha=0.5, label='Testing Data')     # 测试数据
        plt.plot(x_range, y_range_pred, color='green', label=f'Polynomial Degree {degree_best} Fit')  # 模型预测
        plt.title('Polynomial Regression Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

# print(loss_list)

# 创建一个索引列表，用作X轴数据
x_values = np.array(range(1, len(loss_list) + 1))
y_values = np.array(loss_list)

# 创建一个平滑的曲线
x_smooth = np.linspace(x_values.min(), x_values.max(), 300)  # 生成足够多的x值以使曲线平滑
spl = make_interp_spline(x_values, y_values, k=3)  # k是样条曲线的阶数，这里使用三次样条
y_smooth = spl(x_smooth)

# 使用matplotlib绘制平滑的曲线
plt.figure(figsize=[10, 6])
plt.plot(x_smooth, y_smooth, label='Smooth Loss Trend')  # 平滑曲线
plt.title('MAE_loss_trend')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()