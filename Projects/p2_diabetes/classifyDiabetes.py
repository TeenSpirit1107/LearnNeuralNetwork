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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

# Training
if __name__ == '__main__':
    EPOCH_NUM = 40 #modification
    for epoch in range(EPOCH_NUM):
        # i是一个epoch中第几次迭代,一共756条数据,每个mini_batch为32,所以一个epoch需要迭代23次
        # data获取的数据为(x,y)
        loss_one_epoch = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss_one_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss_one_epoch / 23)
        epoch_list.append(epoch)
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch + 1, EPOCH_NUM, loss_one_epoch / 23)) # modification

    # Drawing
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
