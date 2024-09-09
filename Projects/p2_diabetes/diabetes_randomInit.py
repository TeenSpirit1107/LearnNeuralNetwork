# This modified version is not for training model, but for showing the random initial loss caused by the random initialization of parameters. In this model, sometimes the randomly initialized model gives a loss similar to the trained model.

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Prepare the dataset
class DiabetesDateset(Dataset):
    # 加载数据集
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, encoding='utf-8')
        self.len = xy.shape[0]  # shape[0]是矩阵的行数,shape[1]是矩阵的列数
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # 获取数据索引
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 获得数据总量
    def __len__(self):
        return self.len


dataset = DiabetesDateset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2,
                          drop_last=True)  # num_workers为多线程


# Define the model
class FNNModel(torch.nn.Module):
    def __init__(self):
        super(FNNModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据的特征有8个,也就是有8个维度,随后将其降维到6维
        self.linear2 = torch.nn.Linear(6, 4)  # 6维降到4维
        self.linear3 = torch.nn.Linear(4, 2)  # 4维降到2维
        self.linear4 = torch.nn.Linear(2, 1)  # 2w维降到1维
        self.sigmoid = torch.nn.Sigmoid()  # 可以视其为网络的一层,而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = FNNModel()

# Define the criterion and optimizer
criterion = torch.nn.BCELoss(reduction='mean')  # 返回损失的平均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # learning rate

test_list = []
loss_list = []

# Training
if __name__ == '__main__':


    # Modified, to show parameter's random initialization

    print("Show Parameter's Random Initialization")
    ITE_NUM = 2
    for i in range(ITE_NUM):
        print("Round %d"%(i+1))
        model = FNNModel()
        print(model.linear1.weight)
        print(model.linear2.weight)
        print(model.linear3.weight)
        print(model.linear4.weight)
        print()

    # Modified, to show the random initial error caused by the random initialization of parameters.
    loss_one_epoch = 0
    TEST_NUM = 10
    for test in range(TEST_NUM):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss_one_epoch += loss.item()
        loss_list.append(loss_one_epoch / 23)
        test_list.append(test)

        model = FNNModel()
        print('Test[{}],loss:{:.6f}'.format(test + 1, loss_one_epoch / 23)) # modification
        loss_one_epoch = 0

    # Drawing
    plt.plot(test_list,loss_list)
    plt.xlabel('test')
    plt.ylabel('loss')
    plt.show()
