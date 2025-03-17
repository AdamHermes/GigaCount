import torch
import os
from typing import Tuple, Optional, Any, Union
import json
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
from models.clip.ModifiedResnet import ModifiedResNet
from models.clip.image_encoder import ModifiedConvNextMultiScale
from models.clip.feature_fusion import FeatureFusion
from models.clip.text_encoder import CLIPTextEncoder
from models.clip.utils import tokenize
def _resnet(
    reduction: int = 32,
    features_only: bool = False,
    out_indices: Optional[Tuple[int, ...]] = None
) -> ModifiedResNet:
    # with open(os.path.join(curr_dir, "configs", f"clip_image_encoder_{name}.json"), "r") as f:
    #     config = json.load(f)
    model = ModifiedResNet(
        layers=[3,4,6,3],
        output_dim=1024,
        input_resolution=224,
        width=64,
        heads=32,
        features_only=features_only,
        out_indices=out_indices,
        reduction=reduction
    )
    # state_dict = torch.load(os.path.join(curr_dir, "weights", f"clip_image_encoder_{name}.pth"), map_location="cpu")
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # if len(missing_keys) > 0 or len(unexpected_keys) > 0:
    #     print(f"Missing keys: {missing_keys}")
    #     print(f"Unexpected keys: {unexpected_keys}")
    # else:
    #     print(f"All keys matched successfully.")

    return model

ResNetmodel = _resnet(features_only=True,out_indices=(-1,))
ConvNextmodel = ModifiedConvNextMultiScale(
            base_width=96,
            layers=[3, 3, 9, 3],
            output_dim=512,
            input_resolution=224,
        )
FeatureFuse = FeatureFusion(in_channels_list=[96, 192, 384], out_channels=512)
ResNetmodel.eval()
ConvNextmodel.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
num_to_word = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", 
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", 
    "20": "twenty", "21": "twenty-one", "22": "twenty-two", "23": "twenty-three", "24": "twenty-four", "25": "twenty-five", "26": "twenty-six", "27": "twenty-seven", "28": "twenty-eight", "29": "twenty-nine",
    "30": "thirty", "31": "thirty-one", "32": "thirty-two", "33": "thirty-three", "34": "thirty-four", "35": "thirty-five", "36": "thirty-six", "37": "thirty-seven", "38": "thirty-eight", "39": "thirty-nine",
    "40": "forty", "41": "forty-one", "42": "forty-two", "43": "forty-three", "44": "forty-four", "45": "forty-five", "46": "forty-six", "47": "forty-seven", "48": "forty-eight", "49": "forty-nine",
    "50": "fifty", "51": "fifty-one", "52": "fifty-two", "53": "fifty-three", "54": "fifty-four", "55": "fifty-five", "56": "fifty-six", "57": "fifty-seven", "58": "fifty-eight", "59": "fifty-nine",
    "60": "sixty", "61": "sixty-one", "62": "sixty-two", "63": "sixty-three", "64": "sixty-four", "65": "sixty-five", "66": "sixty-six", "67": "sixty-seven", "68": "sixty-eight", "69": "sixty-nine",
    "70": "seventy", "71": "seventy-one", "72": "seventy-two", "73": "seventy-three", "74": "seventy-four", "75": "seventy-five", "76": "seventy-six", "77": "seventy-seven", "78": "seventy-eight", "79": "seventy-nine",
    "80": "eighty", "81": "eighty-one", "82": "eighty-two", "83": "eighty-three", "84": "eighty-four", "85": "eighty-five", "86": "eighty-six", "87": "eighty-seven", "88": "eighty-eight", "89": "eighty-nine",
    "90": "ninety", "91": "ninety-one", "92": "ninety-two", "93": "ninety-three", "94": "ninety-four", "95": "ninety-five", "96": "ninety-six", "97": "ninety-seven", "98": "ninety-eight", "99": "ninety-nine",
    "100": "one hundred", "200": "two hundred", "300": "three hundred", "400": "four hundred", "500": "five hundred", "600": "six hundred", "700": "seven hundred", "800": "eight hundred", "900": "nine hundred",
    "1000": "one thousand"
}
def num2word(num: Union[int, str]) -> str:
    """
    Convert the input number to the corresponding English word. For example, 1 -> "one", 2 -> "two", etc.
    """
    num = str(int(num))
    return num_to_word.get(num, num)
def format_count(count: Union[float, Tuple[float, float]], prompt_type: str = "word") -> str:
    if count == 0:
        return "There is no person." if prompt_type == "word" else "There is 0 person."
    elif count == 1:
        return "There is one person." if prompt_type == "word" else "There is 1 person."
    elif isinstance(count, (int, float)):
        return f"There are {num2word(int(count))} people." if prompt_type == "word" else f"There are {int(count)} people."
    elif count[1] == float("inf"):
        return f"There are more than {num2word(int(count[0]))} people." if prompt_type == "word" else f"There are more than {int(count[0])} people."
    else:  # count is a tuple of finite numbers
        left, right = count
        left = num2word(left) if prompt_type == "word" else left
        right = num2word(right) if prompt_type == "word" else right
        return f"There are between {left} and {right} people."

def get_text_prompts(bins, prompt_type="word"):
    """
    Generates textual descriptions for given bins.
    
    Args:
        bins (List[Tuple[float, float]]): List of bin ranges.
        prompt_type (str): Type of prompt format ("word" or "number").
    
    Returns:
        List[str]: List of formatted text prompts.  
    """
    formatted_bins = [(b[0], b[1]) if b[0] != b[1] else (b[0], b[0]) for b in bins]
    text_prompts = [format_count(b, prompt_type) for b in formatted_bins]
    print(f"Generated Text Prompts: {text_prompts}")
    return text_prompts
def tokenize_text_prompts(text_prompts):
    """
    Tokenizes a list of text prompts using CLIP's tokenizer.

    Args:
        text_prompts (List[str]): List of text descriptions.

    Returns:
        Tensor: Tokenized text prompts.
    """
    return tokenize(text_prompts)
bins = [
    (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0),
    (4.0, 4.0), (5.0, 5.0), (6.0, 6.0), (7.0, 7.0),
    (8.0, 8.0), (9.0, 9.0), (10.0, 10.0), (11.0, float("inf"))
]
text_prompts = get_text_prompts(bins, prompt_type="word")
tokenized_texts = tokenize_text_prompts(text_prompts)
text_encoder = CLIPTextEncoder(
    embed_dim=1024,
    context_length=77,
    vocab_size=49408,  # Placeholder
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12
)
text_encoder.eval()
# Load and preprocess image
image_path = "./data/qnrf/train/images/0001.jpg"  # Change this to your actual image path
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    resnet_features = ResNetmodel(input_tensor)
    shallow, mid, deep = ConvNextmodel(input_tensor)
    convnext_features = deep.mean(dim=[2, 3])  # Perform Global Average Pooling (B, 384, 14, 14) -> (B, 384)

    
    # Generate text features

    text_features = text_encoder(tokenized_texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Reshape and normalize ResNet features
    resnet_features = resnet_features.view(resnet_features.shape[0], resnet_features.shape[1], -1).mean(dim=-1)  # Global average pooling
    resnet_features = resnet_features / resnet_features.norm(dim=1, keepdim=True)
    projection = nn.Linear(2048, 1024)  # Map 2048 -> 1024
    resnet_features = projection(resnet_features)

    # Reshape and normalize ConvNext features
    projection = nn.Linear(384, 1024)  # Map 384 -> 1024 to match CLIP
    convnext_features = projection(convnext_features)
    convnext_features = convnext_features / convnext_features.norm(dim=1, keepdim=True)


    # Compute similarity scores
    resnet_logits = (resnet_features @ text_features.T).softmax(dim=-1)
    convnext_logits = (convnext_features @ text_features.T).softmax(dim=-1)


# Function to print stats
def print_stats(name, tensor):
    print(f"{name} -> Shape: {tensor.shape}, Mean: {tensor.mean().item():.4f}, Var: {tensor.var().item():.4f}, Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")

print_stats("ResNet50 Features", resnet_features)
print_stats("ModifiedConvNextMultiScale Features", convnext_features)

# Plot probability maps
import math

def visualize_probability_maps(prob_maps, title):
    prob_map = prob_maps.squeeze(0).cpu().numpy()  # Remove batch dim -> (12,)
    
    # Convert 1D probs to 2D heatmap (closest square)
    side = math.ceil(math.sqrt(len(prob_map)))  # Find closest square
    heatmap = np.zeros((side, side))  # Create empty square grid
    heatmap.flat[:len(prob_map)] = prob_map  # Fill with probabilities
    
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title(title)
    plt.show()


visualize_probability_maps(resnet_logits, "ResNet50 Probability Map")
visualize_probability_maps(convnext_logits, "ModifiedConvNextMultiScale Probability Map")
