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

## Usage

To train the model, run:
```sh
python train.py --config configs/qnrf.yaml
```

To evaluate on the test set, run:
```sh
python test.py --model checkpoints/best_model.pth
```

## Results

| Model | MAE | RMSE |
|--------|------|------|
| Baseline | 80.5 | 120.3 |
| Improved Model | 65.2 | 98.7 |

### Example Output:
![Example Detection](assets/example_output.png)

## Contributing

Contributions are welcome!
To contribute:
1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- QNRF Dataset: [https://www.kaggle.com/datasets/qatarmobilityinnovations/qnrf-crowd-counting](https://www.kaggle.com/datasets/qatarmobilityinnovations/qnrf-crowd-counting)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
