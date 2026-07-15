# CRACKMAMBA
Structural crack detection using Vision Mamba synthetic data generation, classification, segmentation, explainability, and Gradio deployment.
# CrackMamba 🏗️
### Structural Crack Detection Using Mamba State Space Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Civil%20Engineering%20AI-brown)
![Architecture](https://img.shields.io/badge/Architecture-Mamba%20SSM-purple)

## The Problem

Structural cracks in bridges, buildings, dams, roads, and tunnels are
the earliest visible sign of failure. Manual inspection is slow,
expensive, inconsistent, and dangerous inspectors must physically
climb structures.

An AI system that can analyse a photograph and:
- Detect every crack pixel precisely
- Measure severity (width, length, area, branching)
- Classify urgency (None / Minor / Moderate / Severe / Critical)
- Run on edge devices at construction sites

...can save lives and reduce maintenance costs by 60–80%.

---

## Why Mamba and Not a CNN or Transformer?

| Property | CNN | Transformer | **Mamba (ours)** |
|---|---|---|---|
| Context range | Local only | Global (O(n²)) | Global (O(n)) |
| Memory | Low | Very high | Low |
| Speed on large images | Fast | Slow | **Fast** |
| Crack detection fit | Poor (cracks are long) | Good but slow | **Ideal** |

Cracks are long, thin, connected structures. A CNN only sees a 3×3
or 5×5 window at a time — it misses the continuity of a crack
spanning 200 pixels. A Transformer sees everything but its memory
grows quadratically (262,144² operations for a 512×512 image).

Mamba processes the image as a sequence with a hidden state that
selectively remembers relevant context — exactly what crack
detection needs — at linear cost.

---

## Original Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **SyntheticCrackGenerator** | Generates unlimited photorealistic crack images + masks from scratch. 4 crack types × 5 surface textures |
| 2 | **CrackAugmentor** | Physics-inspired augmentation: weathering, surface aging, rain streaks, sensor noise — goes far beyond standard flips |
| 3 | **MambaBlock** | Pure PyTorch SSM-inspired block with depthwise conv + GRU-style selective gating. Runs without CUDA mamba-ssm library |
| 4 | **CrackMamba architecture** | Mamba SSM encoder + U-Net ConvBlock decoder + skip connections |
| 5 | **CombinedCrackLoss** | BCE (0.3) + Dice (0.3) + Tversky (0.4) — Tversky prioritises recall (missing a crack is dangerous) |
| 6 | **CrackSeverityIndex (CSI)** | Original composite metric: area + width + length + branching → CSI score 0–100 → severity class |
| 7 | **Ablation study** | Mamba vs Pure CNN U-Net vs Threshold baseline — quantifies Mamba's contribution |
| 8 | **Threshold optimisation** | Full precision-recall-IoU-F1 curve vs threshold — finds optimal operating point |

---

## Architecture

```
Input Image [B, 3, H, W]
        │
        ▼
  ┌─────────────┐
  │    STEM     │  Conv(3→32) + BN + ReLU
  └──────┬──────┘
         │
  ┌──────▼──────┐          skip1 ──────────────────────────────┐
  │  Encoder 1  │  MambaBlock(32)                              │
  └──────┬──────┘                                              │
    stride-2 conv                                              │
  ┌──────▼──────┐      skip2 ──────────────────────┐           │
  │  Encoder 2  │  MambaBlock(64)                  │           │
  └──────┬──────┘                                  │           │
    stride-2 conv                                  │           │
  ┌──────▼──────┐                                  │           │
  │ Bottleneck  │  MambaBlock(128)                 │           │
  └──────┬──────┘                                  │           │
   ConvTranspose                                   │           │
  ┌──────▼──────┐                                  │           │
  │  Decoder 2  │  ConvBlock(256→64) ←──── skip2 ──┘           │
  └──────┬──────┘                                              │
   ConvTranspose                                               │
  ┌──────▼──────┐                                              │
  │  Decoder 1  │  ConvBlock(128→32) ←──────────── skip1 ──────┘
  └──────┬──────┘
         ▼
   Output Head  [B, 1, H, W] → sigmoid → probability map
```

### MambaBlock internals
```
x [B,C,H,W] → reshape [B,L,C] → LayerNorm
    → Linear(C → 2×inner) → split(val, gate)
    → DepthwiseConv1d(val)
    → dt_proj(val_conv) [selectivity Δ]
    → val_conv × sigmoid(gate + Δ)   [selective gating]
    → Linear(inner → C) → reshape [B,C,H,W]
    → + residual
```

---

## CrackSeverityIndex (CSI)

The CSI transforms a binary segmentation mask into an actionable
engineering severity score:

```
crack_area_pct    = crack pixels / total pixels × 100
max_crack_width   = max(distance transform) × 2
total_crack_length= sum(skeletonised mask pixels)
branching_factor  = junction pixels / total_length × 100

CSI = 0.40 × area_component
    + 0.25 × width_component
    + 0.20 × length_component
    + 0.15 × branching_component

CSI range → Severity class:
  0 – 5   → None
  5 – 20  → Minor      (monitor)
  20 – 45 → Moderate   (schedule repair)
  45 – 70 → Severe     (repair within 30 days)
  70 – 100→ Critical   (immediate action required)
```

---

## Results

> *Fill in your numbers after running all cells*

### Test Set Performance

| Metric | Score |
|---|---|
| Accuracy | —% |
| Precision | —% |
| Recall | —% |
| F1 Score | —% |
| IoU (Jaccard) | —% |
| Optimal Threshold | — |

### Ablation Study

| Model | IoU | F1 | Params |
|---|---|---|---|
| Threshold (no ML) | —% | —% | 0 |
| Pure CNN U-Net | —% | —% | — |
| **CrackMamba (ours)** | **—%** | **—%** | **—** |

*Fill in after running CrackMamba_Project.ipynb*

---

## Dataset

This project uses **entirely synthetic data** generated by the original
`SyntheticCrackGenerator` class. No external dataset download required.

| Property | Value |
|---|---|
| Generator | SyntheticCrackGenerator (original) |
| Base images | 400 generated |
| After augmentation | ~1,200 samples |
| Image size | 256 × 256 |
| Crack types | hairline, structural, fatigue, alligator |
| Surface textures | concrete, asphalt, brick, metal, plaster |
| Annotation format | Binary mask (crack=255, background=0) |
| Split | 70% train / 15% val / 15% test |

---

## Project Structure

```
CrackMamba/
│
├── CrackMamba_Project.ipynb   ← Main notebook (run this)
│
├── data/
│   ├── images/                ← Generated crack images
│   ├── masks/                 ← Binary segmentation masks
│   ├── augmented/
│   │   ├── images/
│   │   └── masks/
│   └── metadata.csv           ← Per-image crack statistics
│
├── outputs/
│   ├── models/
│   │   └── crackmamba_best.pth        ← Best checkpoint
│   ├── results/
│   │   ├── training_history.csv
│   │   ├── threshold_analysis.csv
│   │   ├── ablation_study.csv
│   │   └── final_report.json
│   ├── visualisations/
│   │   ├── dataset_samples.png
│   │   ├── augmentation_examples.png
│   │   ├── training_batch.png
│   │   ├── training_curves.png
│   │   ├── evaluation_analysis.png
│   │   ├── crack_severity_analysis.png
│   │   └── ablation_study.png
│   └── reports/
│       └── final_report.json
│
├── app.py                     ← Streamlit deployment app
├── predict.py                 ← CLI inference script
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── .gitignore
└── README.md                  ← This file
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```bash
jupyter notebook CrackMamba_Project.ipynb
```
Run all cells from top to bottom. Dataset is generated automatically.

### 3. Predict on your own image
```bash
python predict.py --image path/to/your/image.jpg --threshold 0.5
```

### 4. Launch web app
```bash
streamlit run app.py
```

---

## Skills Demonstrated

`PyTorch` · `Semantic Segmentation` · `State Space Models (Mamba)` ·
`U-Net Architecture` · `Synthetic Data Generation` · `Image Augmentation` ·
`Dice Loss` · `Tversky Loss` · `IoU / Jaccard Score` · `Ablation Study` ·
`Early Stopping` · `CosineAnnealingLR` · `OpenCV` · `scikit-image` ·
`Distance Transform` · `Skeletonisation` · `CrackSeverityIndex` ·
`Model Checkpointing` · `Threshold Optimisation` · `Research Methodology`

---

## Future Work

- [ ] Train on real crack datasets (SDNET2018, DeepCrack, CFD)
- [ ] Add official Mamba-SSM CUDA kernels for GPU acceleration
- [ ] Integrate Grad-CAM++ visualisation for encoder attention maps
- [ ] Multi-scale detection (detect hairline cracks at higher resolution)
- [ ] Export to ONNX for edge deployment (Raspberry Pi, NVIDIA Jetson)
- [ ] Write as a short research paper targeting CVPR Workshop or Springer LNCS
- [ ] Collect real images from Hyderabad infrastructure for Indian-context validation


---

## Acknowledgements

- Mamba SSM paper: Gu & Dao, "Mamba: Linear-Time Sequence Modeling
  with Selective State Spaces", 2023
- U-Net: Ronneberger et al., MICCAI 2015
- Tversky Loss: Salehi et al., 2017
- Open-source libraries: PyTorch, OpenCV, scikit-image, albumentations

---

*This is an independent capstone project developed as part of Data Science and AI learning journey. I used AI-assisted tools for learning, ideation, and debugging while implementing, understanding, and documenting the complete pipeline myself.*
