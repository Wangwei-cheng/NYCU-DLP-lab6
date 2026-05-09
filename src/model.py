import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ConditionalUnet(nn.Module):
    def __init__(self, num_classes=24, dim=64):
        super().__init__()
        
        # 1. 建立 Hugging Face 的 UNet2DConditionModel
        # 這是目前最主流且強大的架構，支援 Cross-Attention 條件輸入
        self.unet = UNet2DConditionModel(
            sample_size=64,           # 輸入圖片解析度
            in_channels=3,            # RGB
            out_channels=3,           # 預測雜訊也是 RGB
            layers_per_block=2,       # 每個層級有幾個 ResNet Block
            block_out_channels=(64, 128, 256, 512), # 通道數變化
            down_block_types=(
                "DownBlock2D",        # 一般卷積
                "AttnDownBlock2D",    # 帶有 Attention 的卷積
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=512,  # Cross-Attention 的特徵維度
        )
        
        # 2. 條件投影層 (Condition Projection)
        # 將 24 維 Multi-hot 向量轉換成 UNet Cross-Attention 所需的維度
        # 我們將它轉換成 (Batch, 1, 512) 的序列格式
        self.cond_projection = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, time, condition):
        """
        x: (batch, 3, 64, 64) - 帶雜訊的圖片
        time: (batch,) - 時間步 (0 ~ 1000)
        condition: (batch, 24) - Multi-hot 條件向量
        """
        # 將條件投射到 Cross-Attention 空間
        # (batch, 24) -> (batch, 512) -> (batch, 1, 512)
        encoder_hidden_states = self.cond_projection(condition).unsqueeze(1)
        
        # 呼叫 UNet
        # return_tuple=False 會直接回傳預測的雜訊 Tensor
        output = self.unet(
            x, 
            time, 
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        return output
