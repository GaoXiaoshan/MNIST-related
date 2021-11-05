import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image

class MnistDataset(Dataset):
    def __init__(self, file_path):
        self.meta_data = pd.read_csv(file_path)
        self.imgs = self.meta_data.iloc[:, 0]
        self.labels = self.meta_data.iloc[:, 1]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label = self.labels[item]
        try:
            img = np.array(Image.open(img_path))
        except Exception as e:
            img = None
            print(e, img_path)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        return img, label

    def __len__(self):
        return len(self.meta_data)

class lstm(nn.Module):
    def __init__(self, in_dim=28, hidden_dim=64, num_class=10, num_layers=2):
        super(lstm, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(self.in_dim, self.hidden_dim, num_layers)
        self.classifier = nn.Linear(self.hidden_dim, num_class)

    def forward(self, x):
        x = x.squeeze().permute(2,0,1)
        features, _ = self.encoder(x)
        out = self.classifier(features[-1, :, :])
        return out


class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        # self.C =
        # self.BN =
        # self.ReLu =
        # self.P =
        self.layers = [nn.Conv2d(indim, outdim, 3, padding=padding),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.block(x)
        return out

class Conv4(nn.Module):
    def __init__(self, num_fea=64, num_class=10, depth=4):
        super(Conv4, self).__init__()
        blocks = []
        self.final_feat_dim = 64
        self.num_fea = num_fea
        self.num_class = num_class
        for i in range(depth):
            indim = 1 if i == 0 else num_fea
            outdim = 64
            b = ConvBlock(indim, outdim)
            blocks.append(b)
        self.embeding = nn.Sequential(*blocks)
        self.classifier = nn.Linear(self.final_feat_dim, num_class)

    def forward(self, x):
        feature = self.embeding(x)
        feature = feature.view(feature.shape[0], -1)
        out = self.classifier(feature)
        return out

def get_dataloader(file_csv, batch_size, shuffle, num_works):
    print("读取数据：", file_csv)
    data_set = MnistDataset(file_csv)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_works)
    return data_loader


def train_epoch(model, device, train_loader, optimizer, epoch):
    avg_loss = 0
    loss_func = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.squeeze(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx % 5 == 1):
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, batch_idx, len(train_loader), loss.item()))

    print("===> Epoch[{}]: AvgLoss: {:.4f}".format(epoch, avg_loss/len(train_loader)))


def val_epoch(model, device, val_dataloader):
    avg_loss, eval_acc = 0, 0
    loss_func = nn.CrossEntropyLoss()
    data_count = 0

    for batch_idx, (data, target) in enumerate(val_dataloader):
        target = torch.squeeze(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        avg_loss += loss.item()
        _, pred = torch.max(output, 1)
        num_correct = (pred == target).sum()
        eval_acc += num_correct.data.item()
        data_count += len(data)

    print("===> Test: eval_acc: {:.4f}".format(eval_acc / data_count))

    return eval_acc / data_count

def train(train_csv, val_csv):
    train_dataloader = get_dataloader(train_csv, batch_size=256, shuffle=True, num_works=4)
    val_dataloader = get_dataloader(val_csv, batch_size=256, shuffle=False, num_works=4)
    # choose model CNN or LSTM
    model = lstm()
    # model = Conv4()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda:3")
    # model = nn.DataParallel(model, device_ids=[1,2])
    model.to(device)
    print("开始训练")
    max_acc = 0
    for epoch in range(31):
        model.train()
        train_epoch(model, device, train_dataloader, optimizer, epoch)
        if epoch % 5 ==0:
            cur_acc = val_epoch(model, device, val_dataloader)
            max_acc = max(max_acc, cur_acc)

    print("===> Test: max_acc: {:.4f}".format(max_acc))

if __name__ == '__main__':
    train_csv = "/data/gaoshan/MNIST/train.csv"
    val_csv = "/data/gaoshan/MNIST/test.csv"
    train(train_csv, val_csv)