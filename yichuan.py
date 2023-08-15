from deap import base, creator, tools, algorithms
import random
import numpy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import csv

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# 导入数据集
data = pd.read_csv('D:/dl/data/xunlian.csv')

# 数据预处理
scaler = MinMaxScaler()
x = scaler.fit_transform(data.drop('output', axis=1).values)
y = scaler.fit_transform(data['output'].values.reshape(-1, 1))
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

x1 = x[:282]
y1 = y[:282]

x1 = x1.to(device)
y1 = y1.to(device)


# 定义超参数
input_size = 5
output_size = 1
num_epochs = 180
batch_size = 32
k = 4
# 创建DataLoader对象
dataset = torch.utils.data.TensorDataset(x1, y1)

# 定义损失函数和优化器
criterion = nn.SmoothL1Loss()
best_fitness = float('inf')  # 初始化最优适应度为无穷大
best_model = None  # 初始化最优模型
p=0
error=1
# 定义适应度函数
# 定义适应度函数
def fitness(individual):
    global best_fitness, best_model ,p ,error
    hidden_size, learning_rate = individual
    
    # 使用K折交叉验证计算适应度
    kf = KFold(n_splits=k)
    train_loss = []
    
    for train_index, test_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
        # Reset the model
        model = RNN(input_size, int(hidden_size), output_size)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=4.5e-6)

        for epoch in range(num_epochs):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                #前向传播和计算损失
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                train_pred = model(batch_x)
                loss = criterion(train_pred, batch_y)

                # 反向传播和优化器
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录训练集误差
                train_loss.append(loss.item())
                
                
        p=p+1       # 计算x1的误差
        pred1 = scaler.inverse_transform(model(x1).detach().numpy())
        actual1 = scaler.inverse_transform(y1.detach().numpy())
        relative_error1 = np.abs(actual1 - pred1) / actual1
        mean_relative_error1 = np.mean(relative_error1)
        max_error1 = np.max(relative_error1)   

        error=mean_relative_error1 + max_error1 
        if error < best_fitness:
            best_fitness = error
            best_model = model
            torch.save(best_model.state_dict(), 'best_model.pth')
        print(mean_relative_error1,max_error1,p/4)
    return best_fitness,
    
# 创建遗传算法的工具箱
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
logbook = tools.Logbook()
toolbox.register("attr_hidden_size", random.randint, 1, 40)
toolbox.register("attr_learning_rate", random.uniform, 0.00001, 0.01)
toolbox.register("mutate_hidden_size", tools.mutPolynomialBounded, eta=20, low=1, up=40, indpb=0.01)
toolbox.register("mutate_learning_rate", tools.mutPolynomialBounded, eta=20, low=0.00001, up=0.01, indpb=0.01)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_hidden_size, toolbox.attr_learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
def mutate(individual):
    individual[0], = toolbox.mutate_hidden_size([individual[0]])[0]
    individual[1], = toolbox.mutate_learning_rate([individual[1]])[0]
    return individual,
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selRoulette)

# 运行遗传算法
pop = toolbox.population(n=40)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
stats.register("best", numpy.min, axis=0)

for gen in range(25):# 30代
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.4, mutpb=0.2, ngen=1, 
                                   stats=stats, halloffame=hof, verbose=True)
    # 记录当前代的统计信息
    logbook.record(gen=gen, evals=len(pop),lr=hof[0][1], hidden_size=hof[0][0], **stats.compile(pop))
    if hof[0].fitness.values[0] < best_fitness:
        best_fitness = hof[0].fitness.values[0]
        torch.save(best_model.state_dict(), 'best_model_gen_{}.pth'.format(gen))
end_time = time.time()
best_individual = hof[0]
best_fitness = best_individual.fitness.values[0]
avg_fitnesses = logbook.select('avg')
best_individuals = logbook.select('best')
learning_rates = logbook.select('lr')
hidden_sizes = logbook.select('hidden_size')
with open('best_shuju.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['avg_fitnesses', 'best_individuals', 'learning_rates', 'hidden_sizes'])  # 写入列名
    writer.writerows(zip(avg_fitnesses, best_individuals, learning_rates, hidden_sizes))
print('Best individual: ', best_individual)
print('Best fitness: ', best_fitness)
print('Total time: ', (end_time - start_time)/60)