import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import List, Tuple, Union, Optional
from .clip.image_encoder import ModifiedConvNextMultiScale
from .clip.utils import tokenize
from .utils import _init_weights
from .clip.text_encoder import CLIPTextEncoder
from .clip.feature_fusion import FeatureFusion
from .utils import format_count
curr_dir = os.path.abspath(os.path.dirname(__file__))



class GigaCount(nn.Module):
    def __init__(
        self,
        bins: List[Tuple[float, float]],
        anchor_points: List[float],
        input_size: int = 224,
        output_dim: int = 1024,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ModifiedConvNextMultiScale(
            base_width=96,
            layers=[3, 3, 9, 3],
            output_dim=output_dim,
            input_resolution=input_size,
        )
       
        self.feature_fusion = FeatureFusion(in_channels_list=[96, 192, 384], out_channels=output_dim)
        
        self.encoder_reduction = 32
        self.reduction = self.encoder_reduction
        self.channels = 672
        self.clip_embed_dim = output_dim

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
            transformer_layers=6
        )
        state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_text_encoder_resnet50.pth"), map_location="cpu")
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
        x = self.feature_fusion([shallow,mid, deep])
        target_size = (56, 56)  # Match CLIP-EBC
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        x = self.projection(x)

        image_features = x.permute(0, 2, 3, 1)
        text_features = self.text_encoder(self.text_prompts.to(device)) if self.text_features is None else self.text_features.to(device)  # shape (N, C)



        # Normalize safely
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)
        print("Image Features stats: min:", image_features.min().item(), "max:", image_features.max().item(), "std:", image_features.std().item())

        text_features = F.normalize(text_features, p=2, dim =-1)
        print("Text Features stats: min:", self.text_features.min().item(), "max:", self.text_features.max().item(), "std:", self.text_features.std().item())

        if torch.isnan(image_features).any() :
            print("NaN detected in image/text features!")

        logit_scale = self.logit_scale.exp().clamp(max=100)  
        logits = logit_scale * image_features @ text_features.t()
        print("Logits stats: min:", logits.min().item(), "max:", logits.max().item(), "mean:", logits.mean().item())

        logits = logits.permute(0, 3, 1, 2)
        
        

        probs = logits.softmax(dim=1)
        print("Probability Maps", probs.shape)
        print(probs)
        print("Softmax probs stats: min:", probs.min().item(), "max:", probs.max().item())
        print("Anchor Points:", self.anchor_points)


        exp = (probs * self.anchor_points.to(x.device)).sum(dim=1, keepdim=True)
        
        return logits, exp


