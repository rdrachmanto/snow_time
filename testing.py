import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Parameters
IMAGE_SIZE = 512
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
test_dir = './cswv_s6/test'
checkpoint_path = './checkpoints/enhancedgen_s6/generator_epoch_99.pth'  # Replace with the correct checkpoint
output_dir = './../enhancedgen_s6_results'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Dataset class (reuse from training script)
class CloudSnowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        input_image = image.crop((0, 0, IMAGE_SIZE, IMAGE_SIZE))
        target_image = image.crop((IMAGE_SIZE, 0, IMAGE_SIZE * 2, IMAGE_SIZE))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image, img_name  # Also return filename for saving


# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Data loader
test_dataset = CloudSnowDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Generator
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        return self.model(input)

class EdgeEnhancedConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, activation='leaky_relu'):
        super().__init__()

        # Simple "edge" branch:
        # For demonstration, we use a small learnable conv to produce an edge map of size 1 channel.
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Main convolution on the concatenated input (original + edge map => in_channels + 1)
        self.main_conv = nn.Conv2d(in_channels + 1, out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Produce a single-channel edge map
        edge_map = self.edge_conv(x)

        # Concatenate edge map to original input
        # shape: [B, in_channels + 1, H, W]
        x_cat = torch.cat([x, edge_map], dim=1)

        # Main convolution + activation
        out = self.main_conv(x_cat)
        out = self.act(out)
        return out


# simple implementation
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.size()

        # Squeeze for global average pooling
        y = self.avg_pool(x).view(b, c)  # shape: [B, C]

        # Excitation
        y = self.mlp(y).view(b, c, 1, 1)  # shape: [B, C, 1, 1]

        # Scale
        return x * y


# Classic stn
class SpatialTransformer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Localization network (small conv net -> regress 2x3 affine transform)
        # I think adjust kernel sizes
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        # check dimension outputing
        self.fc_loc = nn.Linear(128 * 64 * 64, 6)

        # Init as identity transform
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # x shape: [B, C, H, W]
        # input dependent
        print(f"x: {x.shape}")
        
        xs = self.localization(x)
        print(f"xs: {xs.shape}")
        
        xs = xs.view(xs.size(0), -1)
        print(f"xs after flattening: {xs.shape}")
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)  # [B, 2, 3]

        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # Sample
        x_trans = F.grid_sample(x, grid, align_corners=False)
        return x_trans


# down block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_edge_enhance=True, apply_batchnorm=True, activation='leaky_relu'):
        super().__init__()

        # Optionally replace plain conv with EdgeEnhancedConv
        if use_edge_enhance:
            self.conv = EdgeEnhancedConv(in_channels, out_channels,
                                         kernel_size=4, stride=2, padding=1,
                                         activation=activation)
        else:
            # Standard conv with stride=2
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True) if activation == 'leaky_relu' else nn.ReLU(inplace=True)
            )

        self.bn = nn.BatchNorm2d(out_channels) if apply_batchnorm else nn.Identity()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)  # edge-enhanced or normal conv
        x = self.bn(x)
        x = self.ca(x)  # channel attention
        return x


# upscale block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_edge_enhance=True, apply_batchnorm=True):
        super().__init__()

        # We do transposed convolution to upsample
        self.up_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size=4, stride=2, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels) if apply_batchnorm else nn.Identity()
        self.ca = ChannelAttention(out_channels)

        # After upsampling, we can optionally run edge-enhanced conv for refined features
        if use_edge_enhance:
            self.eec = EdgeEnhancedConv(out_channels, out_channels,
                                        kernel_size=3, stride=1, padding=1, activation='relu')
        else:
            self.eec = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )

    # Originally there is a "skip" param here
    def forward(self, x, skip):
        # x: incoming feature from the previous layer (bottleneck or previous upblock)
        x = torch.cat([x, skip], dim=1)
        # x = torch.cat([x], dim=1)  # shape: [B, in_channels, H, W]

        # 2) Up-sample
        x = self.up_transpose(x)  # shape: [B, out_channels, 2H, 2W]
        x = self.bn(x)

        # 3) Channel Attention
        x = self.ca(x)

        # 4) Edge-Enhanced (or normal) refinement conv
        x = self.eec(x)

        return x


# final UNet
class EnhancedGeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DownBlock(3, 64, use_edge_enhance=True, apply_batchnorm=False)  # first layer often no BN
        self.down2 = DownBlock(64, 128, use_edge_enhance=True)
        self.down3 = DownBlock(128, 256, use_edge_enhance=True)
        self.down4 = DownBlock(256, 512, use_edge_enhance=True)

        # Bottleneck: Spatial Transformer
        # self.spatial_transform = SpatialTransformer(in_channels=512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


        # For skip connections, each UpBlock will see (prev out + skip)
        # So up1 in_channels = 512 (bottleneck) + 512 (skip from down4) => 1024
        self.up1 = UpBlock(in_channels=512 + 512, out_channels=256)
        # up2 in_channels = 256 + 256 => 512
        self.up2 = UpBlock(in_channels=256 + 256, out_channels=128)
        # up3 in_channels = 128 + 128 => 256
        self.up3 = UpBlock(in_channels=128 + 128, out_channels=64)
        # up4 in_channels = 64 + 64 => 128
        self.up4_transpose = nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1)

        # Final activation
        self.final_act = nn.Tanh()

    def forward(self, x):

        d1 = self.down1(x)  # [B,  64, H/2,   W/2]
        d2 = self.down2(d1)  # [B, 128, H/4,   W/4]
        d3 = self.down3(d2)  # [B, 256, H/8,   W/8]
        d4 = self.down4(d3)  # [B, 512, H/16,  W/16]
        
        # Bottleneck transform
        bt = self.bottleneck(d4)

        # 1) UpBlock
        x = self.up1(bt, d4)  # [B, 256, H/8,   W/8]
        # 2) UpBlock
        x = self.up2(x, d3)  # [B, 128, H/4,   W/4]
        # 3) UpBlock
        x = self.up3(x, d2)  # [B,  64, H/2,   W/2]

        # 4) Final up (no separate block here, but we do skip connect with d1)
        x = torch.cat([x, d1], dim=1)  # [B, 64+64=128, H/2, W/2]
        x = self.up4_transpose(x)  # [B, 3, H, W]

        x = self.final_act(x)  # Range -> [-1, 1]
        return x


# Load model
generator = EnhancedGeneratorUNet().to(DEVICE)
generator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
generator.eval()

# Testing and saving results
with torch.no_grad():
    for i, (input_image, target_image, img_name) in enumerate(test_loader):
        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # Generate prediction
        predicted_image = generator(input_image)

        # Move tensors to CPU and convert to numpy arrays
        input_image_np = input_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        predicted_image_np = predicted_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target_image_np = target_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Convert from [-1, 1] to [0, 255] range and to uint8
        input_image_np = ((input_image_np + 1) * 127.5).astype(np.uint8)
        predicted_image_np = ((predicted_image_np + 1) * 127.5).astype(np.uint8)
        target_image_np = ((target_image_np + 1) * 127.5).astype(np.uint8)

        # Convert numpy arrays to PIL images
        input_image_pil = Image.fromarray(input_image_np)
        predicted_image_pil = Image.fromarray(predicted_image_np)
        target_image_pil = Image.fromarray(target_image_np)

        # Create a stitched image
        stitched_image = Image.new('RGB', (IMAGE_SIZE * 3, IMAGE_SIZE))
        stitched_image.paste(input_image_pil, (0, 0))
        stitched_image.paste(predicted_image_pil, (IMAGE_SIZE, 0))
        stitched_image.paste(target_image_pil, (IMAGE_SIZE * 2, 0))

        # Save stitched image
        base_name = os.path.splitext(os.path.basename(img_name[0]))[0]
        stitched_image.save(os.path.join(output_dir, f"{base_name}_stitched.png"))

        print(f"Saved {base_name}_stitched.png")

print("Testing completed and stitched images saved.")
