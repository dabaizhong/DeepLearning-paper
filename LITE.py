"""
Light Inception with boosTing tEchnique
文章主要为解决深度学习模型在TSC问题中存在的训练参数过多、计算资源消耗大而提出
所提出的网络结构只有9814个训练参数
一切得益于三项技术：多路复用、自定义滤波、扩展卷积

"""

class DWSC(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            groups=in_channels, dilation=dilation, padding=dilation * (kernel_size // 2)
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LITEMV(nn.Module):
    """LITEMV 模型"""
    def __init__(self, input_channels = 16, num_classes = 24):
        super().__init__()
        # 第一层：并行多路复用卷积 (Ni = 6)
        self.conv1_multiplex = nn.ModuleList([
            nn.Conv1d(input_channels, 32, kernel_size=k, padding=k // 2) 
            for k in [2, 4, 8, 16, 32, 64]
        ])
        # 第一层：自定义滤波器
        self.conv1_custom = nn.ModuleList([
            nn.Conv1d(input_channels, 32, kernel_size=k, padding=k // 2, bias=False) 
            for k in [2, 4, 8, 16, 32, 64]
        ])
        
        # 第二层和第三层：深度可分离卷积 + 扩展卷积
        self.conv2 = nn.ModuleList([
            DWSC(384, 64, kernel_size=k, dilation=2) 
            for k in [2, 4, 8, 16, 32, 64]
        ])
        self.conv3 = nn.ModuleList([
            DWSC(384, 64, kernel_size=k, dilation=4) 
            for k in [6, 12, 24, 48, 96]
        ])
        
        # 全局平均池化 + 全连接层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(320, num_classes)
        
    def forward(self, x):
        # 调整输入形状为 (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, time_length, variables) -> (batch_size, variables, time_length)
        
        # 第一层：并行多路复用卷积
        out_multiplex = [F.relu(conv(x)) for conv in self.conv1_multiplex]
        out_custom = [conv(x) for conv in self.conv1_custom]  # 不加激活函数，用于特定模式检测
        out1 = torch.cat(out_multiplex + out_custom, dim=1)  # (batch, 384, seq_len)
        
        
        # 第二层：深度可分离卷积
        out2 = torch.cat([F.relu(conv(out1)) for conv in self.conv2], dim=1)  # (batch, 384, seq_len)
        
        # 第三层：深度可分离卷积
        out3 = torch.cat([F.relu(conv(out2)) for conv in self.conv3], dim=1)  # (batch, 64, seq_len)64,320,135
        
        # 全局平均池化 + 全连接层
        out_pool = self.global_pool(out3).squeeze(-1)  # (batch, 64, 1) -> (batch, 64)
        out = self.fc(out_pool)
        return out
