Air Quality Forecasting Using Hybrid Deep Learning Models

CNN–LSTM–Transformer Architecture for Short-Horizon PM2.5 Forecasting

This repository contains the full implementation, training pipeline, and evaluation code for a hybrid deep learning architecture designed for 1-hour-ahead PM2.5 forecasting.
The model integrates Convolutional Neural Networks (CNN) for short-range feature extraction, Long Short-Term Memory (LSTM) networks for medium-range temporal encoding, and Transformer encoders for long-range dependency modelling.

The project was developed as part of the MSc Artificial Intelligence dissertation at Heriot-Watt University.

⸻

1. Project Overview

Air quality forecasting presents challenges due to heterogeneous station behaviour, missing data, and long-range temporal dependencies. This project introduces a leakage-safe, fully reproducible forecasting pipeline featuring:
	•	Causal imputation
	•	Causal lag construction
	•	Per-station normalisation
	•	Strict chronological train-validation-test splits
	•	Multi-scale temporal modelling
	•	Model comparison across CNN-only, LSTM-only, Transformer-only, and the Hybrid model
	•	Extensive evaluation over 175 monitoring stations

The Hybrid model achieves the best overall performance:
	•	RMSE ≈ 18.59
	•	MAE ≈ 10.80
	•	sMAPE ≈ 22.79%
	•	R² ≈ 0.78

  repo/
│
├── a_scripts/                # Execution scripts (run_all.sh, training, inference)
├── b_ingestion/              # Raw CSV → Parquet conversion and cleaning
├── c_preprocess/             # Causal feature engineering, lag windows, scaling
├── d_modelprep/              # Dataset loaders and sequence builders
├── e_training/               # Model definitions, training loops, metrics
│   ├── models/               # CNN, LSTM, Transformer, Hybrid models
│   └── trainers/             # HybridTrainer, baseline trainers
│
├── experiments/
│   ├── artifacts/            # Generated datasets, features, sequences
│   └── results/              # Serialized metrics and evaluation outputs
│
├── configs/                  # YAML configuration files
│   ├── default.yaml
│   ├── mps.yaml              # Mac M-series
│   ├── gpu.yaml              # RunPod / CUDA
│   └── overlay/              # Baseline overrides
│
├── logs/                     # Training logs for each model
├── MP_Overleaf/              # LaTeX sources for dissertation
└── README.md

3. Installation

Python Environment (Recommended)

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Mac M-series:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RunPod / CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. Running the Full Pipeline
bash a_scripts/01_make_data.sh

Step 2: Generate Features and Sequences
bash a_scripts/02_make_features.sh
bash a_scripts/03_make_sequences.sh

Step 3: Train Models
bash a_scripts/04_train_hybrid.sh

Baselines:
bash a_scripts/04_train_cnn_only.sh
bash a_scripts/04_train_lstm_only.sh
bash a_scripts/04_train_transformer_only.sh

All paths and hyperparameters are controlled via YAML configs.

5. Model Architecture Summary

Hybrid Model
	•	3× Conv1D layers (kernels 7, 5, 3)
	•	1× LSTM block (128 units)
	•	3× Transformer encoder layers (multi-head attention)
	•	Global average pooling
	•	1× Linear prediction head

Baselines
	•	CNN-only: convolutional stack + pooling
	•	LSTM-only: single LSTM layer
	•	Transformer-only: encoder stack with self-attention

All models use:
	•	AdamW optimiser
	•	Huber loss
	•	ReduceLROnPlateau scheduler
	•	Batch size = 64
	•	Input window = 96 timesteps
	•	Horizon = 1 hour

⸻

6. Evaluation

Metrics
	•	RMSE, MAE, sMAPE
	•	R²
	•	Per-station analysis
	•	Error distribution curves
	•	Ablation studies
	•	Training-validation diagnostics for each architecture

Key Findings
	•	Hybrid model achieves best overall accuracy.
	•	CNN saturates early and struggles with long-range relationships.
	•	LSTM converges smoothly but underfits high-frequency PM2.5 spikes.
	•	Transformer captures long-range structure but lacks local inductive bias.
	•	Hybrid architecture combines strengths of all three.

⸻

7. Reproducibility

This repository includes:
	•	All YAML configs used in experiments
	•	All training logs
	•	Pipeline scripts
	•	Deterministic random seeds
	•	Station-level normalisation metadata
	•	Dataset statistic summaries
	•	Full dissertation LaTeX source

Every experiment can be reproduced on CPU/GPU/RunPod using the configuration files included.

⸻

8. Citation

If you use this repository or components of the hybrid model, please cite:

Amit Makhija (2025).
Air Quality Forecasting Using Hybrid Deep Learning Models.
MSc Dissertation, Heriot-Watt University.

