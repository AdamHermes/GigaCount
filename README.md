# Crowd Counting with QNRF Dataset

This project implements a deep learning-based crowd counting model using the QNRF dataset. It leverages PyTorch for training and evaluation.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/crowd-counting.git
   cd crowd-counting
   ```

2. Set up a virtual environment:
   ```sh
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

5. Install PyTorch (for CUDA 12.6):
   ```sh
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

6. Install additional packages:
   ```sh
   pip install tqdm
   ```

## Dataset Preparation

1. Download the processed QNRF dataset.
2. Place the dataset inside the `data` folder in the root project.

### Directory Structure:
```
project_root/
│── data/
│   ├── qnrf/
│   │   ├── train/
│   │   ├── test/
│   │   ├── val/
│── src/
│── README.md
```
## Placing Pretrained Weights

If you have pretrained `.pth` model weights (inside CLIP-EBC\models\clip\_clip\weights), place them inside the `models/weights/` folder (only use the weights for clip_text_encoder_resnet50.pth).

### Example:
Ensure the structure is as follows:
```
project_root/
│── models/
│   ├── weights/
│   │   ├── clip_text_encoder_resnet50.pth
```
## Usage

To train the model, run:
```sh
python train.py
```

