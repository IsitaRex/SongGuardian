from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 *5*4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.linear(x)
        pred = F.softmax(x, dim=1)
        return pred

if __name__ == '__main__':
    model = CNN()
    print(model)
    summary(model, (1,64, 44))

    #summary(model, (1, 128, 13459))