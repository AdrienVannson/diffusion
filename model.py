import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        ks = 3 # Kernel size

        sizes = [32, 32, 64, 64, 128]

        # Block 1
        self.conv1_1 = torch.nn.Conv2d(3, sizes[0], ks, padding='same')
        self.conv1_2 = torch.nn.Conv2d(sizes[0], sizes[0], ks, padding='same')

        # Block 2
        self.conv2_1 = torch.nn.Conv2d(sizes[0], sizes[1], ks, padding='same')
        self.conv2_2 = torch.nn.Conv2d(sizes[1], sizes[1], ks, padding='same')

        # Block 3
        self.conv3_1 = torch.nn.Conv2d(sizes[1], sizes[2], ks, padding='same')
        self.conv3_2 = torch.nn.Conv2d(sizes[2], sizes[2], ks, padding='same')

        # Block 4
        self.conv4_1 = torch.nn.Conv2d(sizes[2], sizes[3], ks, padding='same')
        self.conv4_2 = torch.nn.Conv2d(sizes[3], sizes[3], ks, padding='same')
        
        # Middle block
        self.conv5_1 = torch.nn.Conv2d(sizes[3], sizes[4], ks, padding='same')
        self.conv5_2 = torch.nn.Conv2d(sizes[4], sizes[4], ks, padding='same')

        # Block 4
        self.conv6_up = nn.ConvTranspose2d(sizes[4], sizes[3], kernel_size=2, stride=2)
        self.conv6_1 = torch.nn.Conv2d(2*sizes[3], sizes[3], ks, padding='same')
        self.conv6_2 = torch.nn.Conv2d(sizes[3], sizes[3], ks, padding='same')
        
        # Block 3
        self.conv7_up = nn.ConvTranspose2d(sizes[3], sizes[2], kernel_size=2, stride=2)
        self.conv7_1 = torch.nn.Conv2d(2*sizes[2], sizes[2], ks, padding='same')
        self.conv7_2 = torch.nn.Conv2d(sizes[2], sizes[2], ks, padding='same')
        
        # Block 2
        self.conv8_up = nn.ConvTranspose2d(sizes[2], sizes[1], kernel_size=2, stride=2)
        self.conv8_1 = torch.nn.Conv2d(2*sizes[1], sizes[1], ks, padding='same')
        self.conv8_2 = torch.nn.Conv2d(sizes[1], sizes[1], ks, padding='same')
        
        # Block 1
        self.conv9_up = nn.ConvTranspose2d(sizes[1], sizes[0], kernel_size=2, stride=2)
        self.conv9_1 = torch.nn.Conv2d(2*sizes[0], sizes[0], ks, padding='same')
        self.conv9_2 = torch.nn.Conv2d(sizes[0], sizes[0], ks, padding='same')

        # ToDo 4: Output Part of Network.
        self.conv10 = torch.nn.Conv2d(sizes[0], 3, ks, padding='same')

        # Other layers
        self.maxpooling = torch.nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Block 1
        x1 = self.relu(self.conv1_1(x))
        x1 = self.relu(self.conv1_2(x1))
        
        # Block 2
        x2 = self.maxpooling(x1)
        x2 = self.relu(self.conv2_1(x2))
        x2 = self.relu(self.conv2_2(x2))
        
        # Block 3
        x3 = self.maxpooling(x2)
        x3 = self.relu(self.conv3_1(x3))
        x3 = self.relu(self.conv3_2(x3))
        
        # Block 4
        x4 = self.maxpooling(x3)
        x4 = self.relu(self.conv4_1(x4))
        x4 = self.relu(self.conv4_2(x4))

        # Middle block
        x5 = self.maxpooling(x4)
        x5 = self.relu(self.conv5_1(x5))
        x5 = self.relu(self.conv5_2(x5))

        # Block 4
        x4 = torch.cat((self.conv6_up(x5), x4), 1)
        x4 = self.relu(self.conv6_1(x4))
        x4 = self.relu(self.conv6_2(x4))

        # Block 3
        x3 = torch.cat((self.conv7_up(x4), x3), 1)
        x3 = self.relu(self.conv7_1(x3))
        x3 = self.relu(self.conv7_2(x3))

        # Block 2
        x2 = torch.cat((self.conv8_up(x3), x2), 1)
        x2 = self.relu(self.conv8_1(x2))
        x2 = self.relu(self.conv8_2(x2))

        # Block 1
        x1 = torch.cat((self.conv9_up(x2), x1), 1)
        x1 = self.relu(self.conv9_1(x1))
        x1 = self.relu(self.conv9_2(x1))

        output = self.conv10(x1)

        return output
