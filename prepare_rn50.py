import os
import json
import urllib.request
import torch
from torch import nn, Tensor
from typing import Union
from tqdm import tqdm
from models.clip.CLIP import build_model
# RN50 Model URL (CLIP)
MODEL_URL = "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
MODEL_NAME = "RN50"
MODEL_SAVE_NAME = "resnet50"

# Paths
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join("models/", "weights")
CONFIG_DIR = os.path.join(CURR_DIR, "configs")
MODEL_PATH = os.path.join(WEIGHT_DIR, "RN50.pt")

# Ensure directories exist
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Function to download model if not present
def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading {MODEL_URL} to {MODEL_PATH}...")
    with urllib.request.urlopen(MODEL_URL) as response, open(MODEL_PATH, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        chunk_size = 8192
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as progress:
            while True:
                buffer = response.read(chunk_size)
                if not buffer:
                    break
                out_file.write(buffer)
                progress.update(len(buffer))
    print("Download complete!")
    return MODEL_PATH

# Load model function (based on CLIP)
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in MODEL_NAME:
        model_path = download_model(MODEL_NAME, download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model

# CLIP Text Encoder (Temporary Wrapper)
class CLIPTextEncoderTemp(nn.Module):
    def __init__(self, clip: nn.Module):
        super().__init__()
        self.context_length = clip.context_length
        self.vocab_size = clip.vocab_size
        self.dtype = clip.dtype
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

    def forward(self, text: Tensor) -> None:
        pass

# Prepare RN50 Model
def prepare_rn50():
    print("Preparing CLIP RN50 model...")
    device = torch.device("cpu")
    
    # Load the full model
    model = load(device=device).to(device)

    # Extract encoders
    image_encoder = model.visual.to(device)
    text_encoder = CLIPTextEncoderTemp(model).to(device)

    # Save weights separately
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, f"clip_{MODEL_SAVE_NAME}.pth"))
    torch.save(image_encoder.state_dict(), os.path.join(WEIGHT_DIR, f"clip_image_encoder_{MODEL_SAVE_NAME}.pth"))
    torch.save(text_encoder.state_dict(), os.path.join(WEIGHT_DIR, f"clip_text_encoder_{MODEL_SAVE_NAME}.pth"))

    # Save configuration files
    model_config = {
        "embed_dim": model.embed_dim,
        "image_resolution": model.image_resolution,
        "vision_layers": model.vision_layers,
        "vision_width": model.vision_width,
        "vision_patch_size": model.vision_patch_size,
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
        "transformer_width": model.transformer_width,
        "transformer_heads": model.transformer_heads,
        "transformer_layers": model.transformer_layers,
    }
    image_encoder_config = {
        "embed_dim": model.embed_dim,
        "image_resolution": model.image_resolution,
        "vision_layers": model.vision_layers,
        "vision_width": model.vision_width,
        "vision_patch_size": model.vision_patch_size,
        "vision_heads": model.vision_heads,
    }
    text_encoder_config = {
        "embed_dim": model.embed_dim,
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
        "transformer_width": model.transformer_width,
        "transformer_heads": model.transformer_heads,
        "transformer_layers": model.transformer_layers,
    }

    # Write configs
    with open(os.path.join(CONFIG_DIR, f"clip_{MODEL_SAVE_NAME}.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(CONFIG_DIR, f"clip_image_encoder_{MODEL_SAVE_NAME}.json"), "w") as f:
        json.dump(image_encoder_config, f, indent=4)
    with open(os.path.join(CONFIG_DIR, f"clip_text_encoder_{MODEL_SAVE_NAME}.json"), "w") as f:
        json.dump(text_encoder_config, f, indent=4)

    print("RN50 model preparation complete!")



# Run the preparation function
if __name__ == "__main__":
    prepare_rn50()
