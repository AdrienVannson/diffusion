import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Process the time
        self.time1 = nn.Linear(1, 64)
        self.time2 = nn.Linear(64, 32)

        ks = 3 # Kernel size

        sizes = [128, 128, 128, 128]

        # Block 1
        self.conv1_1 = nn.Conv2d(3 + 32, sizes[0], ks, padding='same')
        self.conv1_2 = nn.Conv2d(sizes[0], sizes[0], ks, padding='same')

        # Block 2
        self.conv2_1 = nn.Conv2d(sizes[0] + 32, sizes[1], ks, padding='same')
        self.conv2_2 = nn.Conv2d(sizes[1], sizes[1], ks, padding='same')

        # Block 3
        self.conv3_1 = nn.Conv2d(sizes[1] + 32, sizes[2], ks, padding='same')
        self.conv3_2 = nn.Conv2d(sizes[2], sizes[2], ks, padding='same')

        # Block 4
        self.conv4_1 = nn.Conv2d(sizes[2] + 32, sizes[3], ks, padding='same')
        self.conv4_2 = nn.Conv2d(sizes[3], sizes[3], ks, padding='same')
        
        # Middle block
        self.linear1 = nn.Linear(128*16 + 32, 64*16)
        self.linear2 = nn.Linear(64*16 + 32, 128*16)

        # Block 4
        self.conv6_1 = nn.Conv2d(sizes[3] + 32, sizes[3], ks, padding='same')
        self.conv6_2 = nn.Conv2d(sizes[3], sizes[3], ks, padding='same')
        
        # Block 3
        self.conv7_up = nn.ConvTranspose2d(sizes[3], sizes[2], kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(sizes[2] + 32, sizes[2], ks, padding='same')
        self.conv7_2 = nn.Conv2d(sizes[2], sizes[2], ks, padding='same')
        
        # Block 2
        self.conv8_up = nn.ConvTranspose2d(sizes[2], sizes[1], kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(sizes[1] + 32, sizes[1], ks, padding='same')
        self.conv8_2 = nn.Conv2d(sizes[1], sizes[1], ks, padding='same')
        
        # Block 1
        self.conv9_up = nn.ConvTranspose2d(sizes[1], sizes[0], kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(sizes[0] + 32, sizes[0], ks, padding='same')
        self.conv9_2 = nn.Conv2d(sizes[0], sizes[0], ks, padding='same')

        # Output
        self.conv10 = nn.Conv2d(sizes[0], 3, ks, padding='same')

        # Other layers
        self.maxpooling = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # Weights initialization
        """for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)"""

    def forward(self, x, t):
        # Process the time
        ts = self.relu(self.time1(t.view(len(t), 1)))
        ts = self.relu(self.time2(ts))

        # Block 1
        x = torch.cat((x, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)), dim=1)
        x1 = self.relu(self.conv1_1(x))
        x1 = self.relu(self.conv1_2(x1))
        
        # Block 2
        x2 = self.maxpooling(x1)
        x2 = torch.cat((x2, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)), dim=1)
        x2 = self.relu(self.conv2_1(x2))
        x2 = self.relu(self.conv2_2(x2))
        
        # Block 3
        x3 = self.maxpooling(x2)
        x3 = torch.cat((x3, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)), dim=1)
        x3 = self.relu(self.conv3_1(x3))
        x3 = self.relu(self.conv3_2(x3))
        
        # Block 4
        x4 = self.maxpooling(x3)
        x4 = torch.cat((x4, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)), dim=1)
        x4 = self.relu(self.conv4_1(x4))
        x4 = self.relu(self.conv4_2(x4))

        # Middle block
        shape = x4.shape
        x_flat = x4.flatten(1, 3)
        x_flat = torch.cat((x_flat, ts), dim=1)
        x_flat = self.relu(self.linear1(x_flat))
        x_flat = torch.cat((x_flat, ts), dim=1)
        x_flat = self.relu(self.linear2(x_flat))

        # Block 4
        x4 = x_flat.view(shape) + x4
        x4 = torch.cat((x4, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)), dim=1)
        x4 = self.relu(self.conv6_1(x4))
        x4 = self.relu(self.conv6_2(x4))

        # Block 3
        x3 = self.conv7_up(x4) + x3
        x3 = torch.cat((x3, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)), dim=1)
        x3 = self.relu(self.conv7_1(x3))
        x3 = self.relu(self.conv7_2(x3))

        # Block 2
        x2 = self.conv8_up(x3) + x2
        x2 = torch.cat((x2, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)), dim=1)
        x2 = self.relu(self.conv8_1(x2))
        x2 = self.relu(self.conv8_2(x2))

        # Block 1
        x1 = self.conv9_up(x2) + x1
        x1 = torch.cat((x1, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)), dim=1)
        x1 = self.relu(self.conv9_1(x1))
        x1 = self.relu(self.conv9_2(x1))

        output = self.conv10(x1)

        return output
