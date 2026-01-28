<div align="center">

# FLAIR #2: Dual-Stream Semantic Segmentation
# VMamba + STmamba Architecture for Multi-Source Optical Imagery






![Static Badge](https://img.shields.io/badge/Code%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/IGNF/FLAIR-1-AI-Challenge/blob/master/LICENSE) <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a> &emsp; ![Static Badge](https://img.shields.io/badge/Dataset%3A-lightgrey?color=lightgrey) [![license](https://img.shields.io/badge/License-IO%202.0-green.svg)](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)



Participate in obtaining more accurate maps for a more comprehensive description and a better understanding of our environment! Come push the limits of state-of-the-art semantic segmentation approaches on a large and challenging dataset. Get in touch at ai-challenge@ign.fr


![Alt bandeau FLAIR-IGN](images/flair_bandeau.jpg?raw=true)
<img width="100%" src="images/flair-2_logos-hd.png">
</div>


<div style="border-width:1px; border-style:solid; border-color:#d2db8c; padding-left: 1em; padding-right: 1em; ">


<br>

## 📢 New Repository Available

A more recent FLAIR project is now available here: <br>
👉 [FLAIR-HUB](https://github.com/IGNF/FLAIR-HUB)
<br><br>


  
<h2 style="margin-top:5px;">Links</h2>


- **Datapaper : https://arxiv.org/pdf/2305.14467.pdf** 

- **Dataset links :** https://ignf.github.io/FLAIR/#FLAIR2

- **Challenge page : https://codalab.lisn.upsaclay.fr/competitions/13447** [🛑 closed!]

</div>
<br><br>





## Context & Data

The FLAIR #2 dataset is sampled countrywide and is composed of over 20 billion annotated pixels of very high resolution aerial imagery at 0.2 m spatial resolution, acquired over three years and different months (spatio-temporal domains). Aerial imagery patches consist of 5 channels (RVB-Near Infrared-Elevation) and have corresponding annotation (with 19 semantic classes or 13 for the baselines). Furthermore, to integrate broader spatial context and temporal information, high resolution Sentinel-2 1-year time series with 10 spectral band are also provided. More than 50,000 Sentinel-2 acquisitions with 10 m spatial resolution are available.
<br>

<p align="center">
  <img width="40%" src="images/flair-2-spatial.png">
  <br>
  <em>Spatial definitions of the FLAIR #2 dataset.</em>
</p>


<p align="center">
  <img width="85%" src="images/flair-2-patches.png">
  <br>
  <em>Example of input data (first three columns are from aerial imagery, fourth from Sentinel-2) and corresponding supervision masks (last column).</em>
</p>

<br><br>
## Architecture

This repository implements a research-ready dual-stream architecture for semantic segmentation:

### Aerial Stream (VMamba)
- Visual State Space Model inspired by VMamba
- Processes 5-channel aerial imagery (RGB + NIR + Elevation)
- Multi-scale feature extraction with 2D selective scan
- 4-stage hierarchical encoder with progressive downsampling

### Satellite Stream (STmamba)
- Spatio-Temporal State Space Model inspired by VideoMamba
- Processes Sentinel-2 temporal sequences (10 bands, variable length)
- Combined spatial and temporal selective scan
- Temporal pooling for feature aggregation

### Fusion Module
- Single-layer cross-attention between last-stage features
- Query from aerial, Key/Value from satellite
- Multi-head attention with 8 heads
- Residual connections and feed-forward network

### Decoder
- Progressive upsampling with skip connections
- Concatenates multi-scale features from both streams
- Produces per-pixel class predictions (13 classes)

<br><br>

## Installation

```bash
# Clone repository
git clone https://github.com/wangyuzhuo116/FLAIR-2.git
cd FLAIR-2

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

The code expects the FLAIR-2 dataset in the following structure:

```
FLAIR-2-main/dataset/
├── train/
│   ├── aerial/
│   │   └── D004_2021/Z1_NN/img/IMG_*.tif
│   ├── labels/
│   │   └── D004_2021/Z1_NN/msk/MSK_*.tif
│   └── sen/
│       └── D004_2021/Z1_NN/sen/SEN2_sp_*.npy
├── test/
│   ├── aerial/
│   ├── labels/
│   └── sen/
├── flair_aerial_metadata.json
└── flair-2_centroids_sp_to_patch.json
```

## Usage

### 1. Build 10k Stratified Subset (Optional)

For rapid experimentation, create a stratified 10k subset:

```bash
python tools/build_subset.py \
    --root-dir ./FLAIR-2-main/dataset \
    --output-dir ./data_splits \
    --subset-size 10000 \
    --val-ratio 0.2 \
    --seed 42
```

**Subset Strategy:**
- Stratified sampling by region (Dxxx_YYYY) and zone (Z1_xx)
- Maintains proportional representation from all regions
- Train:Val split = 8:2 (8000 train, 2000 val)
- Deterministic with fixed seed for reproducibility
- Samples from both train and test to maximize diversity

### 2. Training

Train on full dataset:

```bash
python -m src.cli train --config configs/default.yaml
```

Train on 10k subset:

```bash
python -m src.cli train --config configs/subset_10k.yaml
```

Additional options:

```bash
python -m src.cli train \
    --config configs/default.yaml \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.0001 \
    --resume checkpoints/flair2_dual_stream/last.pth
```

### 3. Evaluation

Evaluate on validation set:

```bash
python -m src.cli eval \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth
```

Evaluate on test set:

```bash
python -m src.cli eval \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth \
    --test
```

### 4. Inference

Run inference and save predictions:

```bash
python -m src.cli inference \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth \
    --output-dir ./predictions
```

### 5. Visualization

Visualize predictions:

```bash
python -m src.cli visualize \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Data paths**: Set `data.root_dir` to your dataset location
- **Model architecture**: Adjust embedding dimensions, depths, etc.
- **Training**: Batch size, learning rate, epochs, augmentation
- **Hardware**: Number of workers, GPU settings
- **Logging**: TensorBoard, CSV logging, checkpoint saving

For subset training, use `configs/subset_10k.yaml` which inherits from `default.yaml`.

## Optimization for L20 GPU

The default configuration is optimized for a single NVIDIA L20 GPU:

- Batch size: 4 (full dataset) / 8 (subset)
- Mixed precision training (AMP) enabled
- Gradient checkpointing in backbones
- Efficient data loading with 4 workers
- Progressive feature downsampling to reduce memory

## Reproducibility

All experiments are reproducible with fixed seed (default: 42):
- Random seed set for Python, NumPy, PyTorch
- CUDNN deterministic mode enabled
- Deterministic subset sampling

## Project Structure

```
FLAIR-2/
├── configs/              # YAML configuration files
│   ├── default.yaml
│   └── subset_10k.yaml
├── src/
│   ├── data/            # Dataset and dataloaders
│   ├── models/          # Model architectures (VMamba, STmamba, fusion, decoder)
│   ├── training/        # Trainer, losses, metrics
│   ├── utils/           # Config, logging, visualization, seed
│   └── cli.py           # Command-line interface
├── tools/
│   └── build_subset.py  # Subset builder tool
├── requirements.txt
└── README.md
```

## Logging and Checkpoints

During training, the following are generated:

- **TensorBoard logs**: `logs/{exp_name}/tensorboard/`
- **CSV logs**: `logs/{exp_name}/metrics.csv`
- **Checkpoints**: `checkpoints/{exp_name}/`
  - `best.pth`: Best model based on validation mIoU
  - `last.pth`: Last epoch checkpoint
  - `epoch_N.pth`: Periodic checkpoints every 10 epochs

## Metrics

The following metrics are computed:
- **mIoU**: Mean Intersection over Union (primary metric)
- **Accuracy**: Pixel-wise accuracy
- **F1 Score**: Per-class and mean F1 score

## Ablation Studies

The architecture supports ablation studies:

1. **Aerial-only**: Set `model.decoder.channels` to use only aerial features
2. **Satellite-only**: Modify fusion to use only satellite stream
3. **Fusion variants**: Switch between cross-attention and simple concatenation
4. **Backbone depths**: Adjust `model.aerial.depths` and `model.satellite.depths`

## Citation

Please cite the FLAIR-2 dataset paper:

```bibtex
@inproceedings{ign2023flair2,
      title={FLAIR: a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery}, 
      author={Anatol Garioud and Nicolas Gonthier and Loic Landrieu and Apolline De Wit and Marion Valette and Marc Poupée and Sébastien Giordano and Boris Wattrelos},
      year={2023},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS) 2023},
      doi={https://doi.org/10.48550/arXiv.2310.13336},
}
```


<br><br><br>

## Reference
Please include a citation to the following article if you use the FLAIR #2 dataset:

```bibtex
@inproceedings{ign2023flair2,
      title={FLAIR: a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery}, 
      author={Anatol Garioud and Nicolas Gonthier and Loic Landrieu and Apolline De Wit and Marion Valette and Marc Poupée and Sébastien Giordano and Boris Wattrelos},
      year={2023},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS) 2023},
      doi={https://doi.org/10.48550/arXiv.2310.13336},
}
```

## Acknowledgment
This work was performed using HPC/AI resources from GENCI-IDRIS (Grant 2022-A0131013803). This work was supported by the project "Copernicus / FPCUP” of the European Union, by the French Space Agency (CNES) and by Connect by CNES.<br>





## Dataset license

The "OPEN LICENCE 2.0/LICENCE OUVERTE" is a license created by the French government specifically for the purpose of facilitating the dissemination of open data by public administration. 
If you are looking for an English version of this license, you can find it on the official GitHub page at the [official github page](https://github.com/etalab/licence-ouverte).

As stated by the license :

### Applicable legislation

This licence is governed by French law.

### Compatibility of this licence

This licence has been designed to be compatible with any free licence that at least requires an acknowledgement of authorship, and specifically with the previous version of this licence as well as with the following licences: United Kingdom’s “Open Government Licence” (OGL), Creative Commons’ “Creative Commons Attribution” (CC-BY) and Open Knowledge Foundation’s “Open Data Commons Attribution” (ODC-BY).
