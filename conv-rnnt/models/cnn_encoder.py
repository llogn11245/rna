import torch
import torch.nn as nn
import torch.nn.functional as F

# class LocalCNNEncoder(nn.Module):
#     def __init__(self, channels=[100, 100, 64, 64], kernel_size=5):
#         super().__init__()
#         self.conv_layers = nn.ModuleList()
#         padding = (kernel_size - 1, 0)  # Causal padding for time dimension
        
#         for i in range(4):
#             self.conv_layers.append(nn.Sequential(
#                 nn.Conv2d(1, channels[i], kernel_size=(kernel_size, kernel_size),
#                           stride=1, padding=padding),
#                 # nn.BatchNorm2d(channels[i]),
#                 nn.ReLU(),
#                 nn.ConstantPad2d((0, 0, 0, -padding[0]), 0)  # Trim padding
#             ))

#     def forward(self, x):
#         """
#         Apply local CNN layers to the input tensor.
#         Args:
#             x (torch.Tensor): Input tensor of shape [B, T, F].
#         Returns:
#             torch.Tensor: Output tensor after applying local CNN layers -> shape [B, T, C] with C: number of channels.
#         """
#         # x: [B, T, F] (eg. [B, 100, 80])
#         x = x.unsqueeze(1)  # Add channel dim [B, 1, T, F] (eg. [B, 1, 100, 80])
#         for layer in self.conv_layers:
#             x = layer(x)
#         x = x.squeeze(1).transpose(1, 2)  # [B, T, C]
#         return x

class LocalCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=[100,100,64,64], kernel_size=5, freq_kernel=5):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, out_ch in enumerate(channels):
            conv = nn.Conv2d(
                in_channels,
                out_ch,
                kernel_size=(kernel_size, freq_kernel),
                stride=(1,1),
                padding=(0,0),
                bias=False
            )
            self.layers.append(nn.Sequential(conv, nn.ReLU()))
            in_channels = out_ch

    def forward(self, x):
        """
        Args:
            x: [B, T, F]  (B=batch, T=time, F=frequency)
        Output: 
            [B, T, C]  (C=channels)
        """
        x = x.unsqueeze(1) # [B, 1, T, F]
        # Causal padding for time dimension
        pad_t = self.layers[0][0].kernel_size[0] - 1  # kernel_size – 1
        for layer in self.layers:
            # F.pad pad = (w_left, w_right, h_top, h_bottom)
            x = F.pad(x, (0, 0, pad_t, 0))  
            x = layer(x)                   # conv + ReLU
        # [B, T, C]
        x = x.squeeze(1).transpose(1, 2)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.shape
        y = x.mean(dim=2)  # Global average pooling
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class GlobalCNNBlock(nn.Module):
    def __init__(self, input_dim, dilation=1):
        super().__init__()
        self.pw1 = nn.Conv1d(input_dim, 2*input_dim, 1)
        self.dw = nn.Conv1d(2*input_dim, 2*input_dim, 3, 
                           padding=dilation, dilation=dilation, 
                           groups=2*input_dim)
        self.bn = nn.BatchNorm1d(2*input_dim)
        self.se = SqueezeExcitation(2*input_dim)
        self.pw2 = nn.Conv1d(2*input_dim, input_dim, 1)
        self.res = nn.Conv1d(input_dim, input_dim, 1) if input_dim != 2*input_dim else None
        
    def forward(self, x):
        residual = x
        x = self.pw1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.bn(self.dw(x.transpose(1, 2)).transpose(1, 2)))
        x = self.se(x.transpose(1, 2)).transpose(1, 2)
        x = self.pw2(x.transpose(1, 2)).transpose(1, 2)
        if self.res:
            residual = self.res(residual.transpose(1, 2)).transpose(1, 2)
        return F.relu(x + residual)

class GlobalCNNEncoder(nn.Module):
    def __init__(self, input_dim, num_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 3)  # Cyclic dilation pattern
            self.blocks.append(GlobalCNNBlock(input_dim, dilation))
            
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def build_cnn_encoder(config):
    local_cnn = LocalCNNEncoder(
        input_dim=config["input_dim"],
        channels=config.get("local_channels", [256, 256, 256, 256]),
        kernel_size=config.get("local_kernel_size", 5)
    )
    
    global_cnn = GlobalCNNEncoder(
        input_dim=config["local_channels"][-1],
        num_blocks=config.get("num_global_blocks", 6)
    )
    
    projection = nn.Linear(
        config["local_channels"][-1], 
        config["output_dim"]
    )
    
    return nn.Sequential(local_cnn, global_cnn, projection)

# Usage example in config:
# cnn_config = {
#     "input_dim": 80,
#     "local_channels": [256, 256, 256, 256],
#     "num_global_blocks": 6,
#     "output_dim": 512
# }
# cnn_encoder = build_cnn_encoder(cnn_config)

import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=0, 
            dilation=dilation,
            groups=groups
        )
    
    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, 0))  # Pad left
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class GlobalCNNEncoder(nn.Module):
    def __init__(self, input_channels=40, num_filters=256, se_reduction=16):
        super().__init__()
        # Layer 1: Point-wise Causal Conv1D
        self.conv1 = CausalConv1d(input_channels, num_filters, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Layer 2: Dilated Depth-wise Causal Conv1D
        self.conv2 = CausalConv1d(
            num_filters, 
            num_filters, 
            kernel_size=3, 
            dilation=2, 
            groups=num_filters  # Depth-wise
        )
        self.bn2 = nn.BatchNorm1d(num_filters)
        
        # Layer 3: Point-wise Causal Conv1D
        self.conv3 = CausalConv1d(num_filters, num_filters * 2, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 2)
        
        # SE Block
        self.se = SEBlock(num_filters * 2, reduction=se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Layer 1
        x = self.relu(self.bn1(self.conv1(x)))
        # Layer 2
        x = self.relu(self.bn2(self.conv2(x)))
        # Layer 3
        x = self.relu(self.bn3(self.conv3(x)))
        # SE Block
        x = self.se(x)
        return x

# Example usage
model = GlobalCNNEncoder(input_channels=40)
input_tensor = torch.randn(32, 40, 100)  # (batch, channels, time_steps)
output = model(input_tensor)
print(output.shape)  # Should be (32, 512, 100)