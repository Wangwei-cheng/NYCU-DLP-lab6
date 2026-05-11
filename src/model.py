import torch.nn as nn
from diffusers import UNet2DConditionModel

class ConditionalUnet(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=512,
        )
        
        self.cond_projection = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, time, condition):
        encoder_hidden_states = self.cond_projection(condition).unsqueeze(1)
        
        output = self.unet(
            x, 
            time, 
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        return output
