import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import math


class DigitDataset(Dataset):
    def __init__(self, path):
        xy = np.loadtxt('./data/' + path, delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 1:]
        self.x = np.reshape(self.x, (self.n_samples, 1, 28, 28))
        self.x = torch.from_numpy(self.x)

        
        self.y = torch.from_numpy(xy[:, [0]]).type(torch.int64)
        

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
train_dataset = DigitDataset('Train Data.csv')
test_dataset = DigitDataset('Test Data.csv')


num_epochs = 2
batch_size = 100
num_classes = 10
learning_rate = 0.001
total_samples = train_dataset.n_samples
n_iterartions = math.ceil(total_samples/batch_size)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


for images, labels in train_dataloader:
    print("images.shape: ", images.shape)
    break
class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #output 64x14x14

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #output 256x7x7

            nn.Flatten(), 
            nn.Linear(256*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = NeuralNet(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



n_tot_steps = len(train_dataloader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        outputs = model(images)
        labels = torch.squeeze(labels)

        #print(outputs.dtype)
        #print(labels.dtype)
        

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_tot_steps}, loss = {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_dataloader:
        outputs = model(images)
        labels = torch.squeeze(labels)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        print('nr of total samples')
        print(n_samples)
        n_correct += (predictions == labels).sum().item()
        print('Nr of correct predictions')
        print(n_correct)
    acc = n_correct / n_samples
    print(f'accuracy: {acc}')

