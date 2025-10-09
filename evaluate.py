import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.GigaCount import GigaCount
from dataset import Crowd
from utils_custom import collate_fn
from transforms import ColorJitter, GaussianBlur, RandomResizedCrop, RandomHorizontalFlip, RandomApply, PepperSaltNoise
from torchvision.transforms.v2 import Compose
from eval_metrics import sliding_window_predict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
CHECKPOINT_PATH = "checkpoints/best_model.pth"

def evaluate_model(model=None, checkpoint_path=CHECKPOINT_PATH):
    
    transforms = Compose([
        RandomResizedCrop((512, 512), scale=(0.8, 1.5)),
        #RandomHorizontalFlip(),
        # RandomApply([
        #     ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
        #     GaussianBlur(kernel_size=5, sigma=(0.1, 5.0)),
        #     PepperSaltNoise(saltiness=1e-3, spiciness=1e-3),
        # ], p=(0., 0., 0.)),
    ])
    dataset = Crowd("qnrf", split="val", transforms=None, sigma=None, return_filename=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    if model is None:
        original_bins = {
                "qnrf": {
                    "bins": {
                        "fine": [[0, 0], [1, 1], [2, 2], [3, 3], [4, "inf"]]
                    },
                    "anchor_points": {
                        "fine": {
                            "middle": [0, 1, 2, 3, 4],
                            "average": [0, 1, 2, 3, 4.21937]
                        }
                    }
                }
            }


        bins = original_bins["qnrf"]["bins"]["fine"]
        anchor_points = original_bins["qnrf"]["anchor_points"]["fine"]["average"] 
        bins = [(float(b[0]), float(b[1]) if b[1] != "inf" else float('inf')) for b in bins]
        print("BINS SHAPE: ", bins)
        anchor_points = [float(p) for p in anchor_points]


        model = GigaCount(bins=bins, anchor_points=anchor_points).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint)

    model.eval()
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for batch_idx, (images, labels, density, image_paths) in enumerate(tqdm(dataloader)):

            images, density = images.to(DEVICE), density.to(DEVICE)

            pred_density = sliding_window_predict(model, images, 224, 224) # the window_size and stride = images.size()
            #_, pred_density = model(images)
            pred_counts = pred_density.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist()

            all_preds.extend(pred_counts)
            all_gts.extend([len(gt) for gt in labels])

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)

    mae = np.mean(np.abs(all_preds - all_gts))
    mse = np.mean((all_preds - all_gts) ** 2)
    rmse = np.sqrt(mse)

    print(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

if __name__ == "__main__":
    evaluate_model()
