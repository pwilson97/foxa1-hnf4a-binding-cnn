# FOXA1-HNF4A Binding CNN

Multi-task convolutional neural network for predicting FOXA1 and HNF4A transcription factor binding from DNA sequence, with interpretability analysis (DeepLIFT, TF-MoDISco).

## Overview

This repo contains a deep learning pipeline built on CUT&Tag data from [Hansen et al. (eLife 2022)](https://doi.org/10.7554/eLife.73358), which profiles FOXA1 and HNF4A binding in K562 cells with a dox-inducible expression system (GSE182189).

- **Task**: Binary prediction of FOXA1 binding and HNF4A binding from 1001 bp DNA sequences
- **Architecture**: 3 convolutional layers (shared) → global average pooling → 2 independent sigmoid heads
- **Parameters**: ~218K
- **Performance**: AUROC 0.877 (FOXA1), 0.916 (HNF4A)

## Model architecture

![Multi-Task Binding CNN Architecture](figures/binding_cnn_architecture.png)

## Repository structure

```
foxa1-hnf4a-binding-cnn/
├── src/
│   ├── binding_model.py             # BindingCNN architecture (multi-task)
│   ├── binding_data.py              # Data prep, chr-based splits, DataLoaders
│   ├── binding_train.py             # Training loop, evaluation, ROC/PR curves
│   └── binding_deeplift_modisco.py  # DeepLIFT attributions + TF-MoDISco
├── data/
│   ├── genome/                      # hg19.fa (not tracked)
│   └── raw/                         # narrowPeak files (not tracked)
├── results/
│   ├── peaks/                       # Consensus peaks
│   ├── classified/                  # FP/HP/CB classified BED files
│   ├── sequences/                   # Extracted FASTA sequences
│   └── deep_learning/               # Model checkpoints, attributions, outputs
├── figures/
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Data preparation

The CNN expects pre-processed data from the upstream motif grammar pipeline. You need:

1. **Consensus narrowPeak files** in `results/peaks/`:
   - `FOXA1_alone_FOXA1ab.consensus.narrowPeak`
   - `HNF4A_alone_HNF4Aab.consensus.narrowPeak`
   - `FOXA1_HNF4A_FOXA1ab.consensus.narrowPeak`
   - `FOXA1_HNF4A_HNF4Aab.consensus.narrowPeak`

2. **Classified site BEDs** in `results/classified/`:
   - Sites classified as FP, HP, CB, or both_pioneer

3. **Extracted FASTA sequences** in `results/sequences/`:
   - `FP.fa`, `HP.fa`, `CB.fa`, `both_pioneer.fa` (summit +/- 500 bp from hg19)

4. **Genome FASTA** in `data/genome/`:
   - `hg19.fa` with samtools index

## Usage

### Train the multi-task binding CNN

```bash
python src/binding_train.py
```

Trains with chromosome-based splits (train: chr1-15, val: chr16-19, test: chr20-22+X+Y), early stopping, and BCEWithLogitsLoss with positive class weights. Saves model to `results/deep_learning/binding/binding_model.pt`.

### Run DeepLIFT + TF-MoDISco

```bash
python src/binding_deeplift_modisco.py
```

Computes per-head DeepLIFT attributions (FOXA1 head and HNF4A head separately) using an all-zeros reference, then runs TF-MoDISco to discover motif patterns from each head's importance scores.

## Citation

Data from:

> Hansen JL, Loell KJ, Cohen BA. A test of the pioneer factor hypothesis using ectopic liver gene activation. *eLife*. 2022;11:e73358. doi:10.7554/eLife.73358
