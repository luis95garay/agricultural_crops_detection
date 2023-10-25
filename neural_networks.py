import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class Net(nn.Module):
    def __init__(
            self,
            conv1: int,
            conv2: int,
            conv3: int,
            kernel: int,
            padding: int,
            dropout: float,
            fc1: int
        ):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, conv1, kernel, padding=padding)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(conv1, conv2, kernel, padding=padding)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(conv2, conv3, kernel, padding=padding)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(conv3 * 4 * 4, fc1)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(fc1, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x
