# FLAIR-2 Implementation Guide

## Quick Start

### 1. Installation
```bash
git clone https://github.com/wangyuzhuo116/FLAIR-2.git
cd FLAIR-2
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_implementation.py
```
Expected output: All 4 tests pass ✓

### 3. Run Example
```bash
python example.py
```
This demonstrates the model architecture with dummy data.

## Dataset Setup

### Directory Structure
Place your FLAIR-2 dataset in the following structure:
```
FLAIR-2-main/dataset/
├── train/
│   ├── aerial/D004_2021/Z1_NN/img/*.tif
│   ├── labels/D004_2021/Z1_NN/msk/*.tif
│   └── sen/D004_2021/Z1_NN/sen/*.npy
├── test/
│   ├── aerial/
│   ├── labels/
│   └── sen/
├── flair_aerial_metadata.json
└── flair-2_centroids_sp_to_patch.json
```

### Update Config
Edit `configs/default.yaml`:
```yaml
data:
  root_dir: "./FLAIR-2-main/dataset"
  aerial_metadata: "./FLAIR-2-main/dataset/flair_aerial_metadata.json"
  centroids_file: "./FLAIR-2-main/dataset/flair-2_centroids_sp_to_patch.json"
```

## Training Workflows

### Option 1: Full Dataset Training
```bash
# Direct training
python -m src.cli train --config configs/default.yaml

# With custom settings
python -m src.cli train \
    --config configs/default.yaml \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.0001
```

### Option 2: 10k Subset Training (Recommended for Quick Experimentation)
```bash
# Step 1: Build subset
python tools/build_subset.py \
    --root-dir ./FLAIR-2-main/dataset \
    --output-dir ./data_splits \
    --subset-size 10000 \
    --val-ratio 0.2 \
    --seed 42

# Step 2: Train on subset
python -m src.cli train --config configs/subset_10k.yaml
```

**Subset Strategy:**
- Stratified sampling by region (Dxxx_YYYY) and zone (Z1_xx)
- Maintains proportional diversity across all regions
- 8:2 train/val split (8000/2000)
- Deterministic with seed=42
- Samples from train+test to maximize coverage

## Model Architecture

### Dual-Stream Design
```
Input: Aerial (5 ch) + Sentinel-2 (10 ch × T timesteps)
       ↓                    ↓
    VMamba               STmamba
  (Aerial Stream)    (Satellite Stream)
       ↓                    ↓
    Multi-scale         Multi-scale
    Features            Features
       ↓                    ↓
       └────→ Cross-Attention Fusion ←────┘
                      ↓
            Segmentation Decoder
                      ↓
            Output: 13-class Mask
```

### Component Details

**VMamba Aerial Stream:**
- Input: 5 channels (RGB + NIR + Elevation)
- 4-stage hierarchical encoder
- Dimensions: 96 → 192 → 384 → 768
- Visual state space blocks with 2D selective scan
- Patch embedding with 4×4 patches

**STmamba Satellite Stream:**
- Input: 10 bands × T temporal sequences
- Spatio-temporal state space processing
- Combined spatial + temporal selective scan
- Temporal pooling for feature aggregation
- Same hierarchical structure as aerial

**Fusion Module:**
- Single-layer cross-attention
- Query: Aerial features
- Key/Value: Satellite features
- 8 attention heads, dimension 256
- Residual connections + FFN

**Decoder:**
- Progressive upsampling (4 stages)
- Skip connections from both streams
- Final 4× upsampling to input resolution
- Per-pixel 13-class prediction

### Model Statistics
- Total parameters: ~76M
- Trainable parameters: ~76M
- Default batch size: 4 (L20 GPU optimized)
- Mixed precision: Enabled (AMP)

## Evaluation

### Validate on Val Set
```bash
python -m src.cli eval \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth
```

### Evaluate on Test Set
```bash
python -m src.cli eval \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth \
    --test
```

### Metrics Computed
- **mIoU**: Mean Intersection over Union (primary metric)
- **Accuracy**: Pixel-wise accuracy
- **F1 Score**: Per-class and mean F1

## Inference

### Generate Predictions
```bash
python -m src.cli inference \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth \
    --output-dir ./predictions
```

Output: TIF files with per-pixel class predictions

### Visualize Results
```bash
python -m src.cli visualize \
    --config configs/default.yaml \
    --checkpoint checkpoints/flair2_dual_stream/best.pth
```

Output: PNG visualizations comparing aerial/GT/prediction

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/flair2_dual_stream/tensorboard
```

### CSV Logs
Located at: `logs/flair2_dual_stream/metrics.csv`

Contains epoch-by-epoch metrics for analysis.

### Checkpoints
Saved in: `checkpoints/flair2_dual_stream/`
- `best.pth`: Best validation mIoU
- `last.pth`: Latest epoch
- `epoch_N.pth`: Every 10 epochs

## Configuration Options

### Key Settings (configs/default.yaml)

**Data:**
```yaml
data:
  root_dir: "./FLAIR-2-main/dataset"
  use_subset: false  # true for subset training
```

**Model:**
```yaml
model:
  aerial:
    embed_dim: 96
    depths: [2, 2, 9, 2]
  satellite:
    embed_dim: 96
    depths: [2, 2, 6, 2]
  fusion:
    type: "cross_attention"  # or "simple"
    num_heads: 8
```

**Training:**
```yaml
training:
  num_epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  use_amp: true
  optimizer: "adamw"
  scheduler: "cosine"
```

**Hardware:**
```yaml
hardware:
  num_workers: 4
  pin_memory: true
  seed: 42
```

## Ablation Studies

### Test Different Configurations

**1. Simple Fusion (vs. Cross-Attention):**
```yaml
model:
  fusion:
    type: "simple"
```

**2. Adjust Model Capacity:**
```yaml
model:
  aerial:
    embed_dim: 64  # Smaller model
    depths: [2, 2, 6, 2]
```

**3. Different Loss Functions:**
```yaml
training:
  loss:
    type: "combined"  # CE + Dice
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 2`
- Enable gradient checkpointing (requires code modification)
- Use smaller input resolution

### Slow Training
- Increase `num_workers` in config
- Verify GPU usage: `nvidia-smi`
- Check data loading time vs. compute time

### Poor Convergence
- Try different learning rate: `--lr 0.00005`
- Adjust warmup epochs
- Check class balance (use weighted loss)

## File Structure

```
FLAIR-2/
├── configs/
│   ├── default.yaml          # Full dataset config
│   └── subset_10k.yaml       # Subset config
├── src/
│   ├── data/                 # Dataset & dataloader
│   ├── models/               # Model architectures
│   ├── training/             # Trainer, losses, metrics
│   ├── utils/                # Utilities
│   └── cli.py                # CLI interface
├── tools/
│   └── build_subset.py       # Subset builder
├── test_implementation.py    # Verification tests
├── example.py                # Usage example
├── requirements.txt          # Dependencies
└── README.md                 # Main documentation
```

## Next Steps

1. **Prepare Dataset**: Download and organize FLAIR-2 dataset
2. **Verify Setup**: Run `python test_implementation.py`
3. **Quick Test**: Build subset and train for a few epochs
4. **Full Training**: Train on complete dataset
5. **Evaluation**: Test on held-out data
6. **Inference**: Generate predictions for new data

## Citation

If you use this implementation or the FLAIR-2 dataset, please cite:

```bibtex
@inproceedings{ign2023flair2,
      title={FLAIR: a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery}, 
      author={Anatol Garioud and Nicolas Gonthier and Loic Landrieu and Apolline De Wit and Marion Valette and Marc Poupée and Sébastien Giordano and Boris Wattrelos},
      year={2023},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS) 2023},
      doi={https://doi.org/10.48550/arXiv.2310.13336},
}
```

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the main README.md for additional details
