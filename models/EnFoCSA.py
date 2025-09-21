import torch
import torch.nn as nn
import torch.nn.functional as F

class EnFoCSAModule(nn.Module):
    """
    跨时相空间注意力模块 (Cross Spatial Attention Module, CSAM)
    
    从TinyCD迁移的CSAM模块，用于增强双时相特征图之间的空间关联性。
    通过计算两个时相特征图的空间统计信息（avg和max pooling），
    生成统一的注意力权重并同时应用到两个时相，实现跨时相的空间注意力增强。
    
    Args:
        kernel_size: 卷积核大小，默认为7
    """
    
    def __init__(self, kernel_size=7):
        super(EnFoCSAModule, self).__init__()
        
        # 跨时相注意力生成卷积：4通道输入 → 1通道输出
        # 4通道 = T1的(avg+max) + T2的(avg+max)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
    
    def forward(self, x, y):
        """
        前向传播
        
        Args:
            x: 第一时相特征图 (B, C, H, W)
            y: 第二时相特征图 (B, C, H, W)
            
        Returns:
            Tuple[Tensor, Tensor]: 注意力增强后的两个时相特征图
        """
        # 第一时相的空间统计特征
        x_avg = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        x_attn = torch.cat([x_avg, x_max], dim=1)  # (B, 2, H, W)
        
        # 第二时相的空间统计特征
        y_avg = torch.mean(y, dim=1, keepdim=True)  # (B, 1, H, W)
        y_max = torch.max(y, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        y_attn = torch.cat([y_avg, y_max], dim=1)  # (B, 2, H, W)
        
        # 跨时相特征拼接与注意力权重生成
        attn = torch.cat([x_attn, y_attn], dim=1)  # (B, 4, H, W)
        attn = self.conv(attn)  # (B, 4, H, W) → (B, 1, H, W)
        attn = torch.sigmoid(attn)  # 注意力权重归一化到[0,1]
        
        # 同一注意力权重同时应用到两个时相特征图
        x_enhanced = x * attn  # (B, C, H, W) * (B, 1, H, W) → (B, C, H, W)
        y_enhanced = y * attn  # (B, C, H, W) * (B, 1, H, W) → (B, C, H, W)
        
        return x_enhanced, y_enhanced 