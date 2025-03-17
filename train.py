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
BATCH_SIZE = 8
LEARNING_RATE = 1e-6
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_TYPE = "adaptive_hybrid"
USE_AMP = True  
MAX_BIN_VALUE = 1e6  

transform = Compose([
    RandomResizedCrop((512,512), scale=(0.8,1.5)),
    RandomHorizontalFlip(),
    RandomApply([
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ], p=0.5),  # Apply with 50% probability
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
print("BINS SHAPE: ", bins)
anchor_points = [float(p) for p in anchor_points]
#bins = compute_dynamic_bins(original_bins["qnrf"]["bins"], num_bins=3)
#anchor_points = original_bins["qnrf"]["anchor_points"]["middle"]

model = GigaCount(bins=bins, anchor_points=anchor_points).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

loss_fn = AdaptiveHybridLoss(bins=bins)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scaler = GradScaler("cuda") if USE_AMP else None  

# Load Dataset

def train():
    dataset = Crowd("qnrf", split="train",transforms=transform, sigma=8, return_filename=True)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    data_iter = iter(dataloader)

    model.train()
    for epoch in range(EPOCHS):
        img_id = None
        epoch_loss = 0.0
        all_preds, all_gts = [], []
        num_images = 1201

        for i in range(num_images):
            # Handle StopIteration error by resetting iterator
            if img_id is not None:
                images, labels, density, image_path = dataset[img_id]
            else:
                images, labels, density, image_path = next(data_iter)

            optimizer.zero_grad()
            images, density = images.to(DEVICE), density.to(DEVICE)

            with autocast(enabled=USE_AMP):
                logits, preds = model(images)

                # Debug NaN/Inf values
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("⚠️ NaN/Inf detected in logits!")
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    print("⚠️ NaN/Inf detected in preds!")

                loss, loss_dict = loss_fn(logits, preds, density)

            # Fix list error: Ensure labels is a tensor before calling .detach()
            labels_tensor = torch.cat(labels).detach().cpu()

            gt_counts = labels_tensor.numpy()  
            pred_counts = preds.sum().item()  

            print(f"Ground Truth Counts: {len(gt_counts)}")
            print(f"Predicted Counts: {pred_counts}")

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
            
            all_preds.append(preds.sum().item())  # Store sum of predicted counts for each image
            all_gts.append(len(gt_counts))  # Store number of ground truth objects




            # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)

        # Compute Metrics
        mae = np.mean(np.abs(all_preds - all_gts))  # Mean Absolute Error
        mse = np.mean((all_preds - all_gts) ** 2)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss / len(dataloader):.4f}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
