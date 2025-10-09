import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import List, Tuple, Union, Optional
from .clip.image_encoder import ModifiedConvNextMultiScale, Backbone
from .clip.utils import tokenize
from .utils import _init_weights
from .clip.text_encoder import CLIPTextEncoder
from .clip.feature_fusion import FeatureFusion, ccsm
from .utils import format_count, BasicBlock, make_resnet_layers
curr_dir = os.path.abspath(os.path.dirname(__file__))



class GigaCount(nn.Module):
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        anchor_points: List[float],
        input_size: int = 224,
        output_dim: int = 512,
        freeze_text_encoder: bool = True,
        decoder_cfg = [768],
        decoder_block = BasicBlock
    ) -> None:
        super().__init__()
        # self.image_encoder = ModifiedConvNextMultiScale(
        #     base_width=96,
        #     layers=[3, 3, 9, 3],
        #     output_dim=output_dim,
        #     input_resolution=input_size,
        # )
        self.image_encoder = Backbone()
        num_filters = [  16, 32, 64]
        self.ccsm1 = ccsm(96, 48, num_filters[0])
        self.ccsm2 = ccsm(192, 96, num_filters[1])
        self.ccsm3 = ccsm(384, 192, num_filters[2])

        self.fusion = FeatureFusion(num_filters[0],num_filters[1],num_filters[2])
        

        self.reduction = 8
        self.channels = 112
        self.clip_embed_dim = output_dim
        if decoder_cfg is not None:
            assert decoder_block is not None, "Expected decoder_block to be a nn.Module, got None."
            self.image_decoder = make_resnet_layers(decoder_block, decoder_cfg, in_channels=self.channels, expansion=1, dilation=1)
            self.image_decoder.apply(_init_weights)
            self.channels = decoder_cfg[-1]
        else:
            self.image_decoder = nn.Identity()
        if self.channels != self.clip_embed_dim:

            self.projection = nn.Conv2d(in_channels=self.channels, out_channels=self.clip_embed_dim, kernel_size=1)
            self.projection.apply(_init_weights)
        else:
            self.projection = nn.Identity()

        self.bins = bins
        self.prompt_type = "word"

        self.text_encoder = CLIPTextEncoder(
            embed_dim=output_dim,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
        state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_text_encoder_vit_b_16.pth"), map_location="cpu")
        self.text_encoder.load_state_dict(state_dict, strict=False)
        


        self.freeze_text_encoder = freeze_text_encoder
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        self._get_text_prompts()
        self._tokenize_text_prompts()


        self._extract_text_features()
        

        self.anchor_points = torch.tensor(anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)



    def _get_text_prompts(self) -> None:
        bins = [b[0] if b[0] == b[1] else b for b in self.bins]
        self.text_prompts = [format_count(b, self.prompt_type) for b in bins]



    def _tokenize_text_prompts(self) -> None:
        self.text_prompts = tokenize(self.text_prompts)


    def _extract_text_features(self) -> None:
        with torch.no_grad():
            self.text_features = self.text_encoder(self.text_prompts)


    def forward(self, x: Tensor, true_counts: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:


        device = x.device
        shallow,mid,deep = self.image_encoder(x)
        print("Shallow NaN:", torch.isnan(shallow).any().item())
        print("Mid NaN:", torch.isnan(mid).any().item())
        print("Deep NaN:", torch.isnan(deep).any().item())
        pool1 = self.ccsm1(shallow)
        pool2 = self.ccsm2(mid)
        pool3 = self.ccsm3(deep)
        print("Pool1 NaN:", torch.isnan(pool1).any().item())
        print("Pool2 NaN:", torch.isnan(pool2).any().item())
        print("Pool3 NaN:", torch.isnan(pool3).any().item())
        x = self.fusion(pool1, pool2, pool3)
        
        print("Before Interpolation NaN:", torch.isnan(x).any().item())
        x = F.interpolate(x, size=(28,28), mode="bilinear", align_corners=False)
        print("After Interpolation NaN:", torch.isnan(x).any().item())

        x = self.image_decoder(x)
        print("After Decoder NaN:", torch.isnan(x).any().item())
        x = self.projection(x)
        print("After Projection NaN:", torch.isnan(x).any().item())
        image_features = x.permute(0, 2, 3, 1)
        text_features = self.text_encoder(self.text_prompts.to(device)) if self.text_features is None else self.text_features.to(device)  # shape (N, C)



        # Normalize safely
        image_features = F.normalize(image_features, p=2, dim=-1)
        print("Image Features stats: min:", image_features.min().item(), "max:", image_features.max().item(), "std:", image_features.std().item())

        text_features = F.normalize(text_features, p=2, dim =-1)
        print("Text Features stats: min:", self.text_features.min().item(), "max:", self.text_features.max().item(), "std:", self.text_features.std().item())

        if torch.isnan(image_features).any() :
            print("NaN detected in image/text features!")

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        logits = logits.permute(0, 3, 1, 2)
        
        

        probs = logits.softmax(dim=1)

        print("Softmax probs stats: min:", probs.min().item(), "max:", probs.max().item())


        exp = (probs * self.anchor_points.to(x.device)).sum(dim=1, keepdim=True)
        
        return logits, exp


