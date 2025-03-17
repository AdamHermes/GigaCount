Place the processed qnrf folder in the data folder in the root project
In terminal:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install tqdm

