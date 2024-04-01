import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 数据读取
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')
X_train = train_data['x'].values.reshape(-1, 1)
Y_train = train_data['y'].values.reshape(-1, 1)
X_test = test_data['x'].values.reshape(-1, 1)
Y_test = test_data['y'].values.reshape(-1, 1)

# 定义神经网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(64, 64) # 第二个隐藏层
        self.fc3 = nn.Linear(64, 1)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转换数据为torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()   # 清除之前的梯度
    output = model(X_train_tensor)  # 前向传播
    loss = criterion(output, Y_train_tensor)  # 计算损失
    loss.backward()         # 反向传播，计算梯度
    optimizer.step()        # 更新权重

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 使用训练好的模型进行预测
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    Y_pred_nn = model(X_test_tensor).view(-1).numpy()

# 计算平均绝对误差
mae_nn = mean_absolute_error(Y_test, Y_pred_nn)
print(f'Test MAE of the neural network: {mae_nn}')
