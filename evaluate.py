import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.GigaCount import GigaCount
from dataset import Crowd
from utils import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
CHECKPOINT_PATH = "checkpoints/best_model.pth"

def evaluate_model(model=None, checkpoint_path=CHECKPOINT_PATH):
    """Evaluate the model on the test set and return MAE, RMSE"""
    dataset = Crowd("qnrf", split="val", transforms=None, sigma=None, return_filename=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    if model is None:
        model = GigaCount().to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch_idx, (images, labels, density, image_paths) in enumerate(tqdm(dataloader)):

            images, density = images.to(DEVICE), density.to(DEVICE)

            _, preds = model(images) 


            pred_counts = preds.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist()

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
