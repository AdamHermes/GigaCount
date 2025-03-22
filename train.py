import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from models.GigaCount import GigaCount
from losses.hybrid_loss import AdaptiveHybridLoss
from dataset import Crowd  # Assume dataset module exists
from utils import collate_fn
from eval_metrics import CrowdCountingMetrics
from transforms import ColorJitter, GaussianBlur, RandomResizedCrop, RandomHorizontalFlip,RandomApply,PepperSaltNoise
from torchvision.transforms.v2 import Compose
# Training Configurations
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_TYPE = "adaptive_hybrid"
USE_AMP = True
MAX_BIN_VALUE = 1e6  
transforms = Compose([
    RandomResizedCrop((512, 512), scale=(0.8, 1.5)),
    RandomHorizontalFlip(),
    RandomApply([
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
        GaussianBlur(kernel_size=5, sigma=(0.1, 5.0)),
        PepperSaltNoise(saltiness=1e-3, spiciness=1e-3),
    ], p=(0.2, 0.2, 0.5)),
])
original_bins = {
        "qnrf": {
            "bins": {
                "fine":[
                    [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
                    [5, 5], [6, 6], [7, 7], [8, "inf"]
                ],
                "dynamic": [
                    [0, 0], [1, 1], [2, 2], [3, 3],
                    [4, 5], [6, 7], [8, "inf"]
                ],
                "coarse": [
                    [0, 0], [1, 2], [3, 4], [5, 6], [7, "inf"]
                ]
            },
            "anchor_points": {
                "fine": {
                    "middle": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    "average": [0, 1, 2, 3, 4, 5, 6, 7, 9.23349]
                },
                "dynamic": {
                    "middle": [0, 1, 2, 3, 4.5, 6.5, 8],
                    "average": [0, 1, 2, 3, 4.29278, 6.31441, 9.23349]
                },
                "coarse": {
                    "middle": [0, 1.5, 3.5, 5.5, 7],
                    "average": [0, 1.14978, 3.27641, 5.30609, 8.11466]
                }
            }
        }
    }


bins = original_bins["qnrf"]["bins"]["fine"]
anchor_points = original_bins["qnrf"]["anchor_points"]["fine"]["average"] 
bins = [(float(b[0]), float(b[1]) if b[1] != "inf" else float('inf')) for b in bins]

anchor_points = [float(p) for p in anchor_points]
#bins = compute_dynamic_bins(original_bins["qnrf"]["bins"], num_bins=3)
#anchor_points = original_bins["qnrf"]["anchor_points"]["middle"]

model = GigaCount(bins=bins, anchor_points=anchor_points).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

loss_fn = AdaptiveHybridLoss(bins=bins)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scaler = GradScaler() if USE_AMP else None


# Load Dataset

def train():
    dataset = Crowd("qnrf", split="train", transforms=transforms, sigma=None, return_filename=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model.train()
    best_mae = float('inf')  # Initialize best MAE to a high value
    start_epoch = 0

    # Load checkpoint if available
    checkpoint_path = "checkpoints/latest_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"]) if USE_AMP else None
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        print(f"Resuming from epoch {start_epoch}, Best MAE: {best_mae:.2f}")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        all_preds, all_gts = [], []

        for batch_idx, (images, labels, density, image_paths) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            images, density = images.to(DEVICE), density.to(DEVICE)

            with autocast(enabled=USE_AMP): 
                logits, preds = model(images)
                loss, loss_dict = loss_fn(logits, preds, density)

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()

            labels_tensor = torch.cat(labels).detach().cpu()
            gt_counts = labels_tensor.numpy()
            pred_counts = preds.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist()

            all_preds.extend(pred_counts)  
            all_gts.extend([len(gt) for gt in labels])  


        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)

        mae = np.mean(np.abs(all_preds - all_gts))
        mse = np.mean((all_preds - all_gts) ** 2)
        rmse = np.sqrt(mse)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss / len(dataloader):.4f}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 2 == 0:
            save_checkpoint(epoch, best_mae, model, optimizer, scaler)

        # Save the best model based on MAE
        if mae < best_mae:
            best_mae = mae
            save_best_model(model, best_mae)



def save_checkpoint(epoch, best_mae, model, optimizer, scaler):
    checkpoint = {
        "epoch": epoch,
        "best_mae": best_mae,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
    }
    torch.save(checkpoint, "checkpoints/latest_checkpoint.pth")
    print(f"Checkpoint saved at epoch {epoch+1}")


def save_best_model(model, best_mae):
    torch.save(model.state_dict(), "checkpoints/best_model.pth")
    print(f"Best model saved with MAE: {best_mae:.2f}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()