# import torch
# from torch import nn, Tensor
# import torch.nn.functional as F
# from typing import List, Tuple
# from .blocks import LayerNorm, Transformer

# def compute_dynamic_bins(count_data: List[float], num_bins: int = 3) -> List[Tuple[float, float]]:
#     """
#     Compute dynamic bins using K-Means clustering on the count data.
#     """
#     import numpy as np
#     from sklearn.cluster import KMeans
    
#     count_data = np.array(count_data).reshape(-1, 1)
#     kmeans = KMeans(n_clusters=num_bins, random_state=42)
#     kmeans.fit(count_data)
#     bin_edges = np.sort(kmeans.cluster_centers_.flatten())
#     bin_edges = np.insert(bin_edges, 0, 0)
#     bin_edges = np.append(bin_edges, max(count_data) + 1)
#     return [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges) - 1)]

# def generate_contextual_bin_descriptions(bins: List[Tuple[float, float]]) -> List[str]:
#     """
#     Generates natural language descriptions for each bin range.
#     """
#     descriptions = []
#     for bin_range in bins:
#         if bin_range[1] <= 5:
#             descriptions.append("Very few people present, almost empty space.")
#         elif bin_range[1] <= 15:
#             descriptions.append("Sparse crowd with scattered individuals.")
#         elif bin_range[1] <= 30:
#             descriptions.append("Moderate crowd density, movement is easy.")
#         elif bin_range[1] <= 50:
#             descriptions.append("High crowd density, significant congestion.")
#         else:
#             descriptions.append("Very dense crowd, movement is restricted.")
#     return descriptions

# class CLIPTextEncoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         context_length: int,
#         vocab_size: int,
#         transformer_width: int,
#         transformer_heads: int,
#         transformer_layers: int,
#     ) -> None:
#         super().__init__()
#         self.context_length = context_length
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask(),
#         )
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

#     def build_attention_mask(self):
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # Zero out the lower diagonal
#         return mask
    
#     @property
#     def dtype(self):
#         return self.transformer.resblocks[0].attn.in_proj_weight.dtype

#     def forward(self, text: Tensor):
#         x = self.token_embedding(text).type(self.dtype)
#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
#         return x
import torch
from torch import nn, Tensor

from .blocks import LayerNorm, Transformer


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def forward(self, text: Tensor):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
