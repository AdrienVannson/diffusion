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


class EncodingBlock(nn.Module):
    def __init__(self, nb_features_in, nb_features_time, nb_features_out):
        super().__init__()

        self.conv1 = nn.Conv2d(nb_features_in + nb_features_time, nb_features_out, 3, padding='same')
        self.conv2 = nn.Conv2d(nb_features_out, nb_features_out, 3, padding='same')
        self.batchnorm = nn.BatchNorm2d(nb_features_out)
        self.maxpooling = nn.MaxPool2d(2)
        self.ReLU = nn.ReLU()

    def forward(self, x_in, t):
        t = t.unsqueeze(2).unsqueeze(3).repeat(1, 1, x_in.shape[2], x_in.shape[3])
        x = self.conv1(torch.cat((x_in, t), dim=1))
        x = self.ReLU(x)
        x = self.conv2(x)
        if x.shape[1] == x_in.shape[1]:
            x = x + x_in # Add the residual connection
        x = self.batchnorm(x) # TODO before or after ReLU?
        x = self.ReLU(x)

        return x, self.maxpooling(x)


class DecodingBlock(nn.Module):
    def __init__(self, nb_features_in, nb_features_time, nb_features_out):
        super().__init__()

        self.upsampling = nn.ConvTranspose2d(nb_features_in, nb_features_out, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(nb_features_out + nb_features_time, nb_features_out, 3, padding='same')
        self.conv2 = nn.Conv2d(nb_features_out, nb_features_out, 3, padding='same')
        self.batchnorm = nn.BatchNorm2d(nb_features_out)
        self.ReLU = nn.ReLU()

    def forward(self, x, x_skip, t):
        x = self.upsampling(x)
        x_in = x + x_skip
        t = t.unsqueeze(2).unsqueeze(3).repeat(1, 1, x_in.shape[2], x_in.shape[3])

        x = self.conv1(torch.cat((x_in, t), dim=1))
        x = self.ReLU(x)
        x = self.conv2(x) + x_in # Add the residual connection
        x = self.batchnorm(x) # TODO before or after ReLU?
        x = self.ReLU(x)
        return x


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size

        # Process the time
        self.time1 = nn.Linear(64, 64)
        self.time2 = nn.Linear(64, 64)

        encoder_sizes = [3, 128, 128, 128, 128]
        decoder_sizes = [128, 128, 128, 128, 128]

        # Encoder
        encoder_blocks = []

        for i in range(len(encoder_sizes) - 1):
            s1 = encoder_sizes[i]
            s2 = encoder_sizes[i + 1]

            encoder_blocks.append(EncodingBlock(s1, 64, s2))

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # Middle block
        """s = (input_size // 8) * (input_size // 8)
        self.linear1 = nn.Linear(128*s + 32, 64*s)
        self.bn_middle1 = nn.BatchNorm1d(64*s)
        self.linear2 = nn.Linear(64*s + 32, 128*s)
        self.bn_middle2 = nn.BatchNorm1d(128*s)"""

        # Decoder
        decoder_blocks = []

        for i in range(len(decoder_sizes) - 1):
            s1 = decoder_sizes[i]
            s2 = decoder_sizes[i + 1]

            decoder_blocks.append(DecodingBlock(s1, 64, s2))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        # Output
        self.final_conv = nn.Conv2d(decoder_sizes[-1], 3, 3, padding='same')

        # Other layers
        self.relu = nn.ReLU()

    def forward(self, x, time_enc):
        # Process the time
        t = self.time1(time_enc)
        t = self.relu(t)
        t = self.time2(t) + time_enc # Add a residual connection
        t = self.relu(t)

        # Encoder
        x_stack = []

        for block in self.encoder_blocks:
            x_before_pooling, x = block(x, t)
            x_stack.append(x_before_pooling)

        # Decoder
        for block in self.decoder_blocks:
            x = block(x, x_stack.pop(), t)

        # Middle block
        """shape = x4.shape
        x_flat = x4.flatten(1, 3)
        x_flat = torch.cat((x_flat, ts), dim=1)
        x_flat = self.relu(self.bn_middle1(self.linear1(x_flat)))
        x_flat = torch.cat((x_flat, ts), dim=1)
        x_flat = self.relu(self.bn_middle2(self.linear2(x_flat)))"""

        return self.final_conv(x)
