import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnet import get_encoder
from .EnFoCSA import EnFoCSAModule
from .fusion_modules import TwoStageUniFiREFusion, UniFiREBlock

class EnCANet(nn.Module):
 
    
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 2,
        backbone_name: str = 'tiny',
        pretrained: bool = True,
        backbone_trainable: bool = True,
        **kwargs
    ):
        super().__init__()
        
        if spatial_dims != 2:
            raise ValueError("`spatial_dims` must be 2 for EnCANet.")
            
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.backbone_trainable = backbone_trainable
        
        self.use_csam = True
        self.csam_kernel_size = 7
        self.use_entropy_weighting = True
        self.temporal_fusion_method = "channel_concat"
        self.spatial_fusion_method = "channel_concat"
        
        
        self.encoder = get_encoder(
            backbone_scale=backbone_name,
            freeze_backbone=not backbone_trainable
        )
        
        
        if not backbone_trainable:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        
        self.stageNumber = 4
        self.encoderNameScale = 2
       

        self.backbone_dims = {
            'tiny': [96, 192, 384, 768], 
            'small': [96, 192, 384, 768],
            'base': [128, 256, 512, 1024], 
            'large': [192, 384, 768, 1536],
            'xlarge': [256, 512, 1024, 2048],
            'efficientnetv2_s_22k': [128, 64, 48, 24] 
        }
        
        self.size_dict = {
            'tiny': [24, 96, 192, 384, 768], 
            'small': [24, 96, 192, 384, 768],
            'base': [32, 128, 256, 512, 1024], 
            'large': [48, 192, 384, 768, 1536],
            'xlarge': [64, 256, 512, 1024, 2048],
            'efficientnetv2_s_22k': [32, 24, 48, 64, 128]  
        }
        
        
        self.feature_dims = self.backbone_dims[backbone_name]
        self.size_change = list(reversed(self.size_dict[backbone_name]))
        
        
        self.use_attention = False
        self.use_cbam = False
        self.use_temporal_attention = False
        self.CSAMs = nn.ModuleList()  
        self.FusionBlocks = nn.ModuleList()  
        
        
        self._build_modules()
        
        
        self.ChangeFinalSqueezeConv = UniFiREBlock(
            in_channels=sum(self.size_change[:-1]),  
            out_channels=self.size_change[-1] * self.encoderNameScale,  
            use_se=False,  
            use_residual=True
        )
        
        self.ChangeFinalConv = nn.Sequential(
            UniFiREBlock(
                in_channels=self.size_change[-1] * self.encoderNameScale,  
                out_channels=self.size_change[-1],  
                use_se=False, 
                use_residual=True
            ),
            nn.Conv2d(self.size_change[-1], out_channels, kernel_size=1)  
        )
        

        self.register_hook(self.encoder)
        self.backboneFeatures = []
    
    def _build_modules(self):
        
        for stage in range(self.stageNumber):
            
            csam_module = EnFoCSAModule(kernel_size=self.csam_kernel_size)
            self.CSAMs.append(csam_module)
          
            use_temporal_se = False  
            use_spatial_se = False  
            
            if stage == 0:
               
                fusion_module = TwoStageUniFiREFusion(
                    encoder_channels=self.feature_dims[stage],
                    decoder_channels=None,
                    temporal_fusion_method=self.temporal_fusion_method,
                    spatial_fusion_method=self.spatial_fusion_method,
                    use_temporal_se=use_temporal_se,
                    use_spatial_se=use_spatial_se
                )
            else:
                
                fusion_module = TwoStageUniFiREFusion(
                    encoder_channels=self.feature_dims[stage],
                    decoder_channels=self.size_change[stage-1],
                    temporal_fusion_method=self.temporal_fusion_method,
                    spatial_fusion_method=self.spatial_fusion_method,
                    use_temporal_se=use_temporal_se,
                    use_spatial_se=use_spatial_se
                )
            
            self.FusionBlocks.append(fusion_module)
    
    def register_hook(self, backbone):
        
        def hook(module, input, output):
            self.backboneFeatures.append(output)
        
        #
        if self.backbone_name == 'efficientnetv2_s_22k':
            #
            backbone.stage3.register_forward_hook(hook)  
            backbone.stage2.register_forward_hook(hook)  
            backbone.stage1.register_forward_hook(hook) 
            backbone.stage0.register_forward_hook(hook)  
        else:
         
            for index in range(len(self.feature_dims)):
               
                stage = backbone.stages[index]
                last_block = stage[-1]
                last_block.register_forward_hook(hook)
    
    def apply_entropy_weighting(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
       
        x_flat = x_norm.view(batch_size, channels, -1)
        prob = x_flat * (x_flat > 1e-5).float() 
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)  # 熵计算
        entropy = entropy / 100.0
        weight_adjustment = entropy.unsqueeze(-1).unsqueeze(-1)
        
        
        weight_adjustment = weight_adjustment.expand(-1, -1, x.size(2), x.size(3))
        
        return x * weight_adjustment
    
    def forward(self, x1, x2):
        self.backboneFeatures = []

        
        _ = self.encoder(x1)
        _ = self.encoder(x2)

        
        blocks1 = self.backboneFeatures[0:self.stageNumber]
        blocks2 = self.backboneFeatures[self.stageNumber:]

        self.backboneFeatures = []

        FusionFeatures = []
        change = None

        for stage in range(self.stageNumber):
            eff_last_1 = blocks1.pop()
            eff_last_2 = blocks2.pop()

            
            eff_last_1 = self.apply_entropy_weighting(eff_last_1)
            eff_last_2 = self.apply_entropy_weighting(eff_last_2)

            
            eff_last_1, eff_last_2 = self.CSAMs[stage](eff_last_1, eff_last_2)

            

            if stage == 0:
                change = self.FusionBlocks[stage](
                    x_decoder=None, 
                    skip_T1=eff_last_1, 
                    skip_T2=eff_last_2
                )
            else:
                change = self.FusionBlocks[stage](
                    x_decoder=change, 
                    skip_T1=eff_last_1, 
                    skip_T2=eff_last_2
                )

            FusionFeatures.append(change)

            if stage < self.stageNumber - 1:
                change = F.interpolate(change, scale_factor=2., mode='bilinear', align_corners=True)

        
        for index in range(len(FusionFeatures)):
            scale_factor = 2 ** (self.stageNumber - index - 1)
            FusionFeatures[index] = F.interpolate(
                FusionFeatures[index], scale_factor=scale_factor, 
                mode='bilinear', align_corners=True
            )

        fusion = torch.cat(FusionFeatures, dim=1)

        change = self.ChangeFinalSqueezeConv(fusion)
        change = F.interpolate(change, scale_factor=self.encoderNameScale, 
                               mode='bilinear', align_corners=True)
        change = self.ChangeFinalConv(change)

        return change



def getEnCANet(
    spatial_dims: int = 2,
    in_channels: int = 3,
    out_channels: int = 2,
    backbone_name: str = 'tiny',
    pretrained: bool = True,
    backbone_trainable: bool = True,
    **kwargs
):
    model = EnCANet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        backbone_name=backbone_name,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        **kwargs
    )
    return model






