import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch import flatten
from torch.utils.data import Dataset
import numpy as np
import torch
import json

# CNN


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 48, 5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(48*1*2, 1400)
        self.linrelu1 = nn.ReLU()

        self.lin2 = nn.Linear(1400, 700)

        self.lin3 = nn.Linear(700, 350)

        self.lin4 = nn.Linear(350, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = self.lin1(x)
        x = self.linrelu1(x)

        x = self.lin2(x)
        x = self.linrelu1(x)

        x = self.lin3(x)
        x = self.linrelu1(x)

        x = self.lin4(x)

        return x

# TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# data loader
fp = open('data/clockwise.json')

data = json.load(fp)

frame_index_offset = data[0][3]

# initialize images
images = np.zeros((len(data), 3, 37, 50))
# speed is 0 and turn direction is 1
outputs = np.zeros((len(data), 2, 1))

i = 0
img_x = 0
img_y = 0

print(images[i][0])
print(data[i][0][img_x][img_y][0])

for i in range(0, len(data)):

    data[i][3] -= frame_index_offset
    frame_index = data[i][3]

    # red
    while img_x in range(0, len(images[i][0])):
        while img_y in range(0, len(images[i][0][img_x])):
            images[i][0][img_x][img_y] = data[i][0][img_x][img_y][0]
            img_y = img_y + 1

        img_x = img_x + 1

    img_x = 0
    img_y = 0

    # green
    while img_x in range(0, len(images[i][0])):
        while img_y in range(0, len(images[i][0][img_x])):
            images[i][1][img_x][img_y] = data[i][0][img_x][img_y][1]
            img_y = img_y + 1

        img_x = img_x + 1

    img_x = 0
    img_y = 0

    # blue
    while img_x in range(0, len(images[i][0])):
        while img_y in range(0, len(images[i][0][img_x])):
            images[i][2][img_x][img_y] = data[i][0][img_x][img_y][2]
            img_y = img_y + 1

        img_x = img_x + 1

    # speed
    outputs[i][0] = data[i][1]
    # turn direction
    outputs[i][1] = data[i][2]

images = torch.tensor(images, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# initialize the tcn
n_channels = [1, 5, 2]
k_size = 5
n_inputs = 256

tcn = TemporalConvNet(num_inputs=n_inputs,
                      num_channels=n_channels, kernel_size=k_size).to(device)
cnn = CNN().to(device)

# trainin
batch_size = 60
learning_rate = 0.001
num_epochs = 2000

optimizer = optim.Adam(nn.ModuleList(
    [tcn, cnn]).parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i in range(0, len(images), batch_size):
        images_batch = images[i:i+batch_size].to(device)
        x1 = cnn(images_batch).unsqueeze(-1)
        output_prediction = tcn(x1).squeeze(-1)
        output_batch = outputs[i:i+batch_size].to(device).squeeze(-1)
        loss = loss_fn(output_prediction, output_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')
