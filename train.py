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
from dataset import Crowd # Assume dataset module exists
from utils_custom import collate_fn, cosine_annealing_warm_restarts
from eval_metrics import CrowdCountingMetrics
from transforms import ColorJitter, GaussianBlur, RandomResizedCrop, RandomHorizontalFlip,RandomApply,PepperSaltNoise, NormalizeTensor, CombinedTrainTransform
from torchvision.transforms.v2 import Compose
from evaluate import evaluate_model  # Import evaluation function
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

import torchvision.transforms as T

# Training Configurationsx  
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
EPOCHS = 600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_TYPE = "adaptive_hybrid"
USE_AMP = True  
MAX_BIN_VALUE = 1e6  
CUSTOM_MEAN = [0.485, 0.456, 0.406]
CUSTOM_STD = [0.229, 0.224, 0.225]
transforms = Compose([
    RandomResizedCrop((224,224), scale=(0.8,1.0)),
    RandomHorizontalFlip(),
    RandomApply([
    ColorJitter(brightness=0.1, contrast=0.1, saturation = 0.1),
    GaussianBlur(kernel_size=5, sigma=(0.1, 5.0)),
    PepperSaltNoise(saltiness=1e-3, spiciness=1e-3),
    ], p=(0.2,0.2,0.5)),
])


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
#bins = compute_dynamic_bins(original_bins["qnrf"]["bins"], num_bins=3)
#anchor_points = original_bins["qnrf"]["anchor_points"]["middle"]

model = GigaCount(bins=bins, anchor_points=anchor_points).to(DEVICE)
# model = _clip_ebc(
#     backbone="vit_b_16",
#     input_size=224,
#     reduction=8,
#     bins=bins,
#     anchor_points=anchor_points,
#     prompt_type="word",
#     num_vpt=32,
#     vpt_drop=0.0,
#     deep_vpt=True
# )
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

loss_fn = AdaptiveHybridLoss(bins=bins, reduction=8)




# Scheduler 2: For fine-tuning from epoch 30+

def reset_scheduler(optimizer):
    return LambdaLR(
    optimizer=optimizer,
    lr_lambda=partial(
        cosine_annealing_warm_restarts,
        warmup_epochs=20,      # slightly shorter warmup
        warmup_lr=1e-5,        # not too small, but still gentle
        T_0=80,                # 10 epochs per cycle (first restart after 10)
        T_mult=1,              # double cycle length after each restart
        eta_min=5e-6,          # prevent LR from going too low
        base_lr=LEARNING_RATE          # main suggestion
    )
)
    

# Load Dataset
def train():
    dataset = Crowd("qnrf", split="train", transforms=transforms, sigma=None, return_filename=False,num_crops=2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,collate_fn=collate_fn)
    patience = 20
    no_improve_counter = 0
    model.train()
    best_mae = float('inf') 
    best_rmse = float('inf') # Initialize best MAE to a high value
    start_epoch = 0
    scaler = GradScaler(enabled=USE_AMP) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = reset_scheduler(optimizer)
    # Load checkpoint if available
    checkpoint_path = "checkpoints/latest_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print("got here")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(checkpoint.keys())

        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        best_rmse = checkpoint["best_rmse"]

        # If resuming from early epoch, restore optimizer/scheduler state
     

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = reset_scheduler(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        #if "scheduler_high_state_dict" in checkpoint:
            #scheduler_high.load_state_dict(checkpoint["scheduler_high_state_dict"])
        #if "scheduler_low_state_dict" in checkpoint:
            #scheduler_low = reset_scheduler_low(optimizer)
            #scheduler_low.load_state_dict(checkpoint["scheduler_low_state_dict"])
        if USE_AMP:
            scaler = GradScaler(enabled=USE_AMP)
            scaler.load_state_dict(checkpoint["scaler_state_dict"])



        print(f"Resuming from epoch {start_epoch}, Best MAE: {best_mae:.2f}")
    with open("eval_log.txt", "a") as log:

        for epoch in range(start_epoch, EPOCHS):
            epoch_loss = 0.0
            all_preds, all_gts = [], []

            for batch_idx, (images, labels, density) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                optimizer.zero_grad()
                images, density = images.to(DEVICE), density.to(DEVICE)
                target_points = [p.to(DEVICE) for p in labels]
                with autocast(enabled=USE_AMP): 
                    logits, preds = model(images)
                    loss, loss_info = loss_fn(logits, preds, density, target_points)


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

                pred_counts = preds.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist()

                all_preds.extend(pred_counts)  
                all_gts.extend([len(gt) for gt in labels])  

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']




            print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss / len(dataloader):.4f} - Learning Rate: {current_lr:.6f}")

            log_entry = f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss / len(dataloader):.4f}, Learning Rate: {current_lr:.6f}\n"
            
            print(log_entry)

            with open("eval_log.txt", "a") as log:
                log.write(log_entry)
                log.flush()  # Ensures immediate writing to file
            # Save checkpoint every 5 epochs

            if (epoch + 1) % 2 == 0:
                save_checkpoint(epoch,best_mae,best_rmse,model,optimizer,scaler,scheduler)


            if (epoch + 1) % 5 == 0:

                print(f"Saved checkpoint at epoch {epoch + 1}. Running evaluation...")
                
                mae, rmse = evaluate_model(model)  
                print(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")  


                if mae < best_mae:
                    best_mae = mae
                    no_improve_counter = 0
                    save_best_model(epoch, best_mae, model)
                else:
                    no_improve_counter += 5
                if (rmse < best_rmse):
                    best_rmse = rmse
                    save_best_rmse_model(epoch, best_rmse, model)
                save_checkpoint(epoch,best_mae,best_rmse,model,optimizer,scaler,scheduler)
                # Log evaluation results
                if no_improve_counter >= patience:
                    print("Early stopping triggered.")
                    break
                with open("eval_log.txt", "a") as log:
                    log.write(f"Epoch {epoch + 1} Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")
                    log.flush()
            # Save the best model based on MAE



def save_checkpoint(epoch, best_mae, best_rmse, model, optimizer, scaler, scheduler):
    checkpoint = {
        "epoch": epoch,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
        "scheduler_state_dict": scheduler.state_dict(),

    }
    torch.save(checkpoint, "checkpoints/latest_checkpoint.pth")
    print(f"✅ Checkpoint saved at epoch {epoch+1}")


def save_best_model(epoch, best_mae, model):
    bestmodel = {
        "epoch": epoch,
        "best_mae": best_mae,
        "model_state_dict": model.state_dict(),

    }
    torch.save(bestmodel, "checkpoints/best_model.pth")

    print(f"Best model saved with MAE: {best_mae:.2f}")

def save_best_rmse_model(epoch, best_rmse, model):
    bestmodelrmse = {
        "epoch": epoch,
        "best_rmse": best_rmse,
        "model_state_dict": model.state_dict(),

    }
    torch.save(bestmodelrmse, "checkpoints/best_model_rmse.pth")
    print(f"Best model saved with RMSE: {best_rmse:.2f}")



if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()