"""Regenerate train_sen1floods11.ipynb (Colab/Kaggle) from train_sen1floods11.py."""
import json
from pathlib import Path

here = Path(__file__).parent
script = (here / "train_sen1floods11.py").read_text(encoding="utf-8")


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = [
    md("# Baadhi Watch: train the Sentinel-1 water U-Net (Sen1Floods11)\n"
       "Runtime → Change runtime type → **T4 GPU**, then Run all (~45 min).\n"
       "Copy `water_unet.onnx`, `model_meta.json` and `metrics.json` into `backend/models/`."),
    code("!pip -q install segmentation-models-pytorch rasterio onnx"),
    code("%%writefile train_sen1floods11.py\n" + script),
    code("!python train_sen1floods11.py --download --epochs 40 --batch 16 --workers 2 --out runs/unet"),
    code("from google.colab import files\n"
         "for f in ['water_unet.onnx', 'model_meta.json', 'metrics.json']:\n"
         "    files.download(f'runs/unet/{f}')"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
(here / "train_sen1floods11.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote train_sen1floods11.ipynb")
