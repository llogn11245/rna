import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCNNEncoder(nn.Module):
    def __init__(self, channels=[100, 100, 64, 64], kernel_size=5):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        padding = (kernel_size - 1, 0)  # Causal padding for time dimension
        
        for i in range(4):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(1, channels[i], kernel_size=(kernel_size, 1),
                          stride=1, padding=padding),
                # nn.BatchNorm2d(channels[i]),
                nn.ReLU(),
                nn.ConstantPad2d((0, 0, 0, -padding[0]), 0)  # Trim padding
            ))

    def forward(self, x):
        """
        Apply local CNN layers to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, F].
        Returns:
            torch.Tensor: Output tensor after applying local CNN layers -> shape [B, T, C] with C: number of channels.
        """
        # x: [B, T, F] (eg. [B, 100, 80])
        x = x.unsqueeze(1)  # Add channel dim [B, 1, T, F] (eg. [B, 1, 100, 80])
        for layer in self.conv_layers:
            x = layer(x)
        x = x.squeeze(1).transpose(1, 2)  # [B, T, C]
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