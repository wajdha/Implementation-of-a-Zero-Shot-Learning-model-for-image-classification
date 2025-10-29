# SVAN Zero-Shot Learning - Complete Pipeline

A unified, single-file implementation of the SVAN (Semantic Visual Alignment Network) for Zero-Shot Learning on the Animals with Attributes 2 (AwA2) dataset.

## Quick Start

Run the entire pipeline (data preparation, training, and evaluation) with a single command:

```bash
python svan_full_pipeline.py --backbone resnet50 --lr 0.001 --epochs 10 --plots
```

## Features

✅ **Single executable file**: All modules combined into one script  
✅ **Complete pipeline**: Data preparation → Training → Evaluation → Visualization  
✅ **Multiple backbones**: ResNet-18, ResNet-50, ViT-B/16  
✅ **GZSL evaluation**: Bias calibration with automatic gamma selection  
✅ **Visualization**: Confusion matrix generation  
✅ **ROCm compatible**: Works with AMD GPUs  

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PIL/Pillow

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow
```

## Dataset Setup

1. Download the AwA2 dataset
2. Place it in `~/data/awa2/`
3. Ensure the following structure:
```
~/data/awa2/
├── classes.txt
├── predicates.txt
├── predicate-matrix-continuous.txt
├── trainclasses.txt
├── testclasses.txt
├── JPEGImages/
│   ├── antelope/
│   ├── bear/
│   └── ...
└── ImageSplits/
    ├── train.txt
    ├── val.txt
    ├── test_seen.txt
    └── test_unseen.txt
```

## Usage Examples

### 1. Train and Evaluate (Full Pipeline)
```bash
# ViT-B/16 with default learning rate
python svan_full_pipeline.py --backbone vit_b_16 --lr 0.0001 --epochs 10

# ResNet-50 with higher learning rate and plots
python svan_full_pipeline.py --backbone resnet50 --lr 0.001 --epochs 10 --plots

# ResNet-50 with lower learning rate
python svan_full_pipeline.py --backbone resnet50 --lr 0.00001 --epochs 10
```

### 2. Evaluate Only (Skip Training)
```bash
# Evaluate existing checkpoints
python svan_full_pipeline.py --backbone resnet50 --lr 0.001 --skip_training --plots
```

### 3. Quick Test (Faster)
```bash
# Use ResNet-18 with fewer epochs
python svan_full_pipeline.py --backbone resnet18 --lr 0.0001 --epochs 5 --batch_size 64
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | vit_b_16 | Visual backbone: {vit_b_16, resnet50, resnet18} |
| `--lr` | 0.0001 | Learning rate |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--device` | cuda | Device for evaluation: {cuda, cpu} |
| `--plots` | False | Generate confusion matrix plots |
| `--skip_training` | False | Skip training, only evaluate |

## Output Files

### Checkpoints (in `experiments/` directory)
- `svan_{backbone}_lr{lr}_awa2.pt`: Best model (highest validation accuracy)
- `svan_{backbone}_lr{lr}_awa2_last.pt`: Last epoch model

### Visualizations (if `--plots` flag is used)
- `svan_{backbone}_lr{lr}_awa2_confusion_matrix.png`: Confusion matrix for best model
- `svan_{backbone}_lr{lr}_awa2_last_confusion_matrix.png`: Confusion matrix for last model

## Pipeline Phases

### Phase 1: Data Preparation
- Load AwA2 metadata (50 classes, 85 attributes)
- Create image preprocessing pipelines
- Build PyTorch datasets for train/val/test splits

### Phase 2: Model Initialization
- Initialize SVAN model with specified backbone
- Setup optimizer (Adam) and loss function (CrossEntropy)
- Prepare semantic vectors for seen classes

### Phase 3: Training
- Train model on 40 seen classes
- Validate after each epoch
- Save best and last checkpoints
- Quick ZSL evaluation on 10 unseen classes

### Phase 4: Evaluation
- Load trained checkpoint
- GZSL evaluation with bias calibration (gamma search)
- Generate confusion matrix visualization
- Report metrics: Seen (S), Unseen (U), Harmonic Mean (H)

## Expected Results

### ResNet-50 (lr=0.001, epochs=10)
- Training accuracy: ~91%
- Validation accuracy: ~89%
- ZSL (unseen only): ~46%
- **GZSL Best**: S=87.54%, U=24.91%, H=**38.78%** (gamma=3.0)

### ResNet-50 (lr=0.0001, epochs=10)
- Training accuracy: ~96%
- Validation accuracy: ~95%
- ZSL (unseen only): ~67%
- GZSL Best: S=96.11%, U=12.0%, H=21.33% (gamma=5.0)

### ViT-B/16 (lr=0.0001, epochs=10)
- Training accuracy: ~93%
- Validation accuracy: ~93%
- ZSL (unseen only): ~45%
- GZSL Best: S=93.93%, U=8.9%, H=16.27% (gamma=5.0)

## Hardware Requirements

- **GPU**: NVIDIA (CUDA) or AMD (ROCm)
- **VRAM**: Minimum 8GB (12GB recommended)
- **RAM**: 16GB+
- **Storage**: ~10GB for AwA2 dataset

## Execution Time

On AMD RX 6750 XT GPU:
- Training (10 epochs): ~1.5-2 hours
- Evaluation: ~5-10 minutes
- Total pipeline: ~2 hours

## Troubleshooting

### Error: "Dataset folder not found"
```bash
# Check dataset path
ls ~/data/awa2/
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size
python svan_full_pipeline.py --backbone resnet50 --lr 0.001 --batch_size 16
```

### Error: "Module 'models.svan' not found"
This script is self-contained and doesn't require separate module files. Make sure you're running `svan_full_pipeline.py` directly.

## Code Structure

```python
# Phase 1: Data Preparation Module
- load_awa2_semantics()
- default_transforms()
- AwA2Split class

# Phase 2: Model Architecture Module
- SVAN class (with ResNet/ViT backbones)

# Phase 3: Training Module
- train_model() function

# Phase 4: Evaluation Module
- evaluate_model() function
- acc_with_preds() helper

# Main Pipeline
- main() with argument parsing
```

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{hatem2025zsl,
  title={Implementation of a Zero-Shot Learning Model for Image Classification},
  author={Hatem, Wajd},
  year={2025},
  school={Gdansk University of Technology}
}
```

## License

This implementation is part of a Master's thesis at Gdansk University of Technology.

## Acknowledgments

- AwA2 dataset: Xian et al., "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly"
- Pretrained backbones: torchvision (ImageNet weights)
- ROCm compatibility: AMD GPU support

## Contact

For questions or issues, please contact the author or supervisor.

---

**Master's Thesis Project**  
Gdansk University of Technology  
Faculty of Electrical and Control Engineering  
Field of Study: Automatic Control, Cybernetics and Robotics  
Specialization: Robotics and Decision Systems
