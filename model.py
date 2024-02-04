import torch
from torch import nn

class PositionalEmbedding(nn.Module):

    def __init__(self, max_time, nb_features):
        super().__init__()

        nb_sins = nb_features // 2
        omega = 1 / torch.pow(10_000, torch.arange(nb_sins) / nb_sins).reshape((1, nb_sins))
        positions = torch.arange(0, max_time, dtype=torch.float).reshape((max_time, 1))

        vals = positions @ omega

        # Shape: (token position, embedding dimension)
        self.positional_embedding = torch.cat((torch.sin(vals), torch.cos(vals)), dim=1)

        assert(self.positional_embedding.shape == (max_time, nb_features))

    def forward(self):
        return self.positional_embedding

    def embedding(self):
        return self.positional_embedding


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
        self.bn1 = nn.BatchNorm2d(sizes[0])

        # Block 2
        self.conv2_1 = nn.Conv2d(sizes[0] + 32, sizes[1], ks, padding='same')
        self.conv2_2 = nn.Conv2d(sizes[1], sizes[1], ks, padding='same')
        self.bn2 = nn.BatchNorm2d(sizes[1])

        # Block 3
        self.conv3_1 = nn.Conv2d(sizes[1] + 32, sizes[2], ks, padding='same')
        self.conv3_2 = nn.Conv2d(sizes[2], sizes[2], ks, padding='same')
        self.bn3 = nn.BatchNorm2d(sizes[2])

        # Block 4
        self.conv4_1 = nn.Conv2d(sizes[2] + 32, sizes[3], ks, padding='same')
        self.conv4_2 = nn.Conv2d(sizes[3], sizes[3], ks, padding='same')
        self.bn4 = nn.BatchNorm2d(sizes[3])
        
        # Middle block
        self.linear1 = nn.Linear(128*16 + 32, 64*16)
        self.bn_middle1 = nn.BatchNorm1d(64*16)
        self.linear2 = nn.Linear(64*16 + 32, 128*16)
        self.bn_middle2 = nn.BatchNorm1d(128*16)

        # Block 4
        self.conv6_1 = nn.Conv2d(sizes[3] + 32, sizes[3], ks, padding='same')
        self.conv6_2 = nn.Conv2d(sizes[3], sizes[3], ks, padding='same')
        self.bn6 = nn.BatchNorm2d(sizes[3])
        
        # Block 3
        self.conv7_up = nn.ConvTranspose2d(sizes[3], sizes[2], kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(sizes[2] + 32, sizes[2], ks, padding='same')
        self.conv7_2 = nn.Conv2d(sizes[2], sizes[2], ks, padding='same')
        self.bn7 = nn.BatchNorm2d(sizes[3])
        
        # Block 2
        self.conv8_up = nn.ConvTranspose2d(sizes[2], sizes[1], kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(sizes[1] + 32, sizes[1], ks, padding='same')
        self.conv8_2 = nn.Conv2d(sizes[1], sizes[1], ks, padding='same')
        self.bn8 = nn.BatchNorm2d(sizes[1])
        
        # Block 1
        self.conv9_up = nn.ConvTranspose2d(sizes[1], sizes[0], kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(sizes[0] + 32, sizes[0], ks, padding='same')
        self.conv9_2 = nn.Conv2d(sizes[0], sizes[0], ks, padding='same')
        self.bn9 = nn.BatchNorm2d(sizes[0])

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
        x1 = self.relu(self.bn1(self.conv1_2(x1)))
        
        # Block 2
        x2_in = self.maxpooling(x1)
        x2 = torch.cat((x2_in, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)), dim=1)
        x2 = self.relu(self.conv2_1(x2))
        x2 = self.relu(self.bn2(self.conv2_2(x2) + x2_in))
        
        # Block 3
        x3_in = self.maxpooling(x2)
        x3 = torch.cat((x3_in, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)), dim=1)
        x3 = self.relu(self.conv3_1(x3))
        x3 = self.relu(self.bn3(self.conv3_2(x3) + x3_in))
        
        # Block 4
        x4_in = self.maxpooling(x3)
        x4 = torch.cat((x4_in, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)), dim=1)
        x4 = self.relu(self.conv4_1(x4))
        x4 = self.relu(self.bn4(self.conv4_2(x4) + x4_in))

        # Middle block
        shape = x4.shape
        x_flat = x4.flatten(1, 3)
        x_flat = torch.cat((x_flat, ts), dim=1)
        x_flat = self.relu(self.bn_middle1(self.linear1(x_flat)))
        x_flat = torch.cat((x_flat, ts), dim=1)
        x_flat = self.relu(self.bn_middle2(self.linear2(x_flat)))

        # Block 4
        x4_in = x_flat.view(shape) + x4
        x4 = torch.cat((x4, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)), dim=1)
        x4 = self.relu(self.conv6_1(x4))
        x4 = self.relu(self.bn6(self.conv6_2(x4) + x4_in))

        # Block 3
        x3_in = self.conv7_up(x4) + x3
        x3 = torch.cat((x3, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)), dim=1)
        x3 = self.relu(self.conv7_1(x3))
        x3 = self.relu(self.bn7(self.conv7_2(x3) + x3_in))

        # Block 2
        x2_in = self.conv8_up(x3) + x2
        x2 = torch.cat((x2, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)), dim=1)
        x2 = self.relu(self.conv8_1(x2))
        x2 = self.relu(self.bn8(self.conv8_2(x2) + x2_in))

        # Block 1
        x1_in = self.conv9_up(x2) + x1
        x1 = torch.cat((x1, ts.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)), dim=1)
        x1 = self.relu(self.conv9_1(x1))
        x1 = self.relu(self.bn9(self.conv9_2(x1) + x1_in))

        output = self.conv10(x1)

        return output
