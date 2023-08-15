import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 定义RNN模型

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,  num_layers=2, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐层状态
        h0 = torch.zeros(2, x.size(0), self.hidden_size)
        x = x.unsqueeze(1)
    
        # 前向传播
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out

# 定义超参数
input_size = 5
hidden_size = 22
output_size = 1
num_epochs = 480
learning_rate = 0.006
batch_size = 8
k = 3

# 加载预训练的模型
model = RNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('D:/dl/2/model_22.pth'))


# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只训练第二个RNN层和全连接层的参数
for param in model.rnn.parameters():
    if param.shape[0] == hidden_size:  # 第二个RNN层的参数
        param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# 定义优化器，只优化需要梯度的参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.SmoothL1Loss()

# 导入数据集
data = pd.read_csv('D:/dl/data/xunlian.csv')

# 数据预处理
scaler = MinMaxScaler()
x = scaler.fit_transform(data.drop('output', axis=1).values)
y = scaler.fit_transform(data['output'].values.reshape(-1, 1))
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

train_x = x[283:325]
train_y = y[283:325]

test_x = x[326:372]
test_y = y[326:372]

dataset = torch.utils.data.TensorDataset(train_x, train_y)

# 定义交叉验证器
kf = KFold(n_splits=k)

# 训练模型
train_loss = []
test_loss = []
for train_index, test_index in kf.split(dataset):
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=False) 

    for epoch in range(num_epochs):
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # 前向传播和计算损失
            train_pred = model(batch_x)
            loss = criterion(train_pred, batch_y)

            # 反向传播和优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录训练集误差
            train_loss.append(loss.item())

        # 在测试集上计算误差
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                test_pred = model(batch_x)
                loss = criterion(test_pred, batch_y)
                test_loss.append(loss.item())

        # 打印训练集和测试集误差
        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss[-1], test_loss[-1]))

# 预测结果并绘制图像
with torch.no_grad():
    pred = scaler.inverse_transform(model(train_x).numpy())
    actual = scaler.inverse_transform(train_y.numpy())
    relative_error = np.abs(actual - pred) / actual
    plt.plot(relative_error, label='Relative Error')
    plt.legend()
    plt.show()
with torch.no_grad():
    pred = scaler.inverse_transform(model(train_x).numpy())
    actual =scaler.inverse_transform(train_y.numpy())
    plt.plot(actual, label='Actual')
    plt.plot(pred, label='Predicted')
    plt.legend()
    plt.show()


torch.save(model.state_dict(), 'D:/dl/data/model/qianyi.pth')
relative_error = np.abs(actual - pred) / actual
mean_relative_error = np.mean(relative_error)
max_relative_error = np.max(relative_error)

print('Mean relative error: {:.4f}'.format(mean_relative_error))
print('Max relative error: {:.4f}'.format(max_relative_error))

with torch.no_grad():
    pred = scaler.inverse_transform(model(test_x).cpu().numpy())
    actual = scaler.inverse_transform(test_y.numpy())
    relative_error = np.abs(actual - pred) / actual
    
    mean_relative_error = np.mean(relative_error)
    max_relative_error = np.max(relative_error)


    plt.plot(actual, label='Actual')
    plt.plot(pred, label='Predicted')
    plt.legend()
    plt.show()
    
    plt.plot(relative_error,label='relative_error')
    plt.legend()
    plt.show()

print('Mean relative error: {:.4f}'.format(mean_relative_error))
print('Max relative error: {:.4f}'.format(max_relative_error))
