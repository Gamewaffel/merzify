#!/bin/bash
echo "🇩🇪 Merzify AI - Installation"

# Conda Environment
conda create -n merzify python=3.10 -y
conda activate merzify

# PyTorch mit CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install -r requirements.txt

# InstantID Modelle herunterladen
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
import os

os.makedirs('models/instantid', exist_ok=True)
os.makedirs('models/antelopev2', exist_ok=True)

# InstantID Adapter
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ip-adapter.bin',
    local_dir='models/instantid'
)
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ControlNetModel/config.json',
    local_dir='models/instantid'
)
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ControlNetModel/diffusion_pytorch_model.safetensors',
    local_dir='models/instantid'
)

print('✅ Modelle heruntergeladen!')
"

echo "✅ Installation abgeschlossen! Starte mit: python app.py"
