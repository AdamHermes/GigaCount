# Crowd Counting with QNRF Dataset

This project implements a deep learning-based crowd counting model using the QNRF dataset. It leverages PyTorch for training and evaluation.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AdamHermes/GigaCount.git
   cd GigaCount
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
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

## Dataset Preparation
The link for the original dataset is:
UCF-QNRF: https://www.crcv.ucf.edu/data/ucf-qnrf/

In this repo we will use a preprocessed dataset provided by the paper "CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification."

1. Download the processed QNRF dataset in this drive link:
https://drive.google.com/drive/folders/1X3_MApVSmkRRjHx_a5Tlnv-DUXNWOYnA?usp=sharing
2. Place the dataset inside the `data` folder in the root project.

### Directory Structure:
```
project_root/
│── data/
│   ├── qnrf/
│   │   ├── train/
│   │   ├── val/
│── README.md
```
## Placing Pretrained Weights

The pretrained model weights for the text encoder is provided in the drive link below:
https://drive.google.com/drive/folders/1WjwPZ8jmiiEDl73jpPl8zTMBeot-8j27?usp=sharing

Download and place the weights inside the `models/weights/` folder (only use the weights for clip_text_encoder_resnet50.pth).

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

