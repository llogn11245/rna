import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCNNBlock(nn.Module):
    def __init__(self):
        super(LocalCNNBlock, self).__init__()
        # Không sử dụng padding mặc định trong Conv2d, sẽ pad thủ công
        self.conv1 = nn.Conv2d(1, 100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 100, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(100, 64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, T, F]
        # Padding: (pad_left, pad_right, pad_top, pad_bottom) cho F và T
        x = F.pad(x, (2, 2, 4, 0))  # pad_left=4, pad_right=0 cho T; pad=2 cho F
        x = self.relu(self.conv1(x))

        x = F.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv2(x))

        x = F.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv3(x))

        x = F.pad(x, (2, 2, 4, 0))
        x = self.relu(self.conv4(x))

        return x  # [B, 64, T, F]

# Thử nghiệm
block = LocalCNNBlock()
input_data = torch.randn(1, 1, 100, 80)  # Ví dụ: B=1, T=100, F=80
output = block(input_data)
print(output.shape)  # Kết quả: [1, 64, 100, 80]