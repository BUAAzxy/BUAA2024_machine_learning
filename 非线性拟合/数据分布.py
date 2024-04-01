import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel("test.xlsx", usecols=[0, 1])

# 提取x和y
x = df.iloc[:, 0]
y = df.iloc[:, 1]

# 绘制散点图
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('test')
plt.grid(True)
plt.show()
