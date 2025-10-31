README.md (drop-in)

Hybrid CNN-LSTM-Transformer for Air Quality Forecasting (PyTorch)

End-to-end pipeline to forecast PM2.5 using a hybrid deep learning model:
	•	CNN for local temporal patterns
	•	LSTM for sequential memory
	•	Transformer (encoder-only) for long-range dependencies
The repo contains ingestion → feature engineering → sequence windowing → training → evaluation, wired by shell scripts.

Repo Structure

a_scripts/                 # Orchestrator scripts (run_all.sh, per-stage scripts)
b_ingestion/               # CSV→Parquet, stream assembly, dataset health checks
c_dataprep/                # Imputation, feature engineering, scaling, locking features
d_modelprep/               # Sequence windowing (lookback, horizon, shards)
e_training/                # Model, losses, device utils, train & eval runners
configs/                   # default.yaml, keep.yaml (station selection), etc.
experiments/               # artifacts, models, reports (created at runtime)
raw_data/                  # your input CSVs (ignored by git)

Data Requirements

Place these CSVs under raw_data/:
	•	airquality.csv, meteorology.csv, station.csv, district.csv, city.csv, weatherforecast.csv

The repo ignores raw_data/ and experiments/ via .gitignore to keep the repo lightweight.

Quickstart 
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # or: pip install torch torchvision torchaudio pyarrow numpy pandas pyyaml scikit-learn tqdm
export OMP_NUM_THREADS=8
CONFIG="configs/default.yaml,configs/keep.yaml" bash a_scripts/run_all.sh


Quickstart (CUDA – A100/A5000 on RunPod/Colab/Any Linux GPU)

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Example for CUDA 12.1 wheels (adjust if your image differs):
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt  # rest of deps
export OMP_NUM_THREADS=8
CONFIG="configs/default.yaml,configs/keep.yaml" bash a_scripts/run_all.sh

Configuration

All knobs are in configs/default.yaml:
	•	sequence.lookback, sequence.horizon
	•	model.hybrid.* (CNN channels/kernels, d_model, nhead, num_layers, lstm_hidden, etc.)
	•	dl.batch_size, epochs, patience, amp (mixed precision), optimizer/scheduler
	•	Station filtering via configs/keep.yaml

  Outputs
	•	Artifacts: experiments/artifacts/
	•	features_scaled_ps/ (scalers, splits)
	•	seq/ (sharded windows)
	•	models_*/ (checkpoints, train summaries)
	•	reports/ (eval_report.json, per_station_H*.csv, predictions CSVs)

Reproducibility
	•	Global seed set in default.yaml (seed: 42)
	•	Feature list is locked at experiments/artifacts/features_locked/feature_list.json

Performance Tips
	•	On A100/A5000: keep amp.enabled: true, use dtype: fp16 or bf16 (A100 supports both).
	•	Increase dl.batch_size until you near GPU memory limit.
	•	Tuning levers that matter most:
	•	model.hybrid.d_model, nhead, num_layers
	•	sequence.lookback, horizon
	•	dl.lr, epochs, patience, weight_decay
	•	If IO-bound, raise dl.num_workers (typically 4–8 per GPU) and set pin_memory: true.

Troubleshooting
	•	CUDA OOM: lower dl.batch_size first; then reduce d_model or num_layers.
	•	MPS OOM (Mac): reduce batch size; set prefetch_factor: 0..2; avoid high worker counts.
	•	Slow dataloading: ensure num_workers > 0, shard windows (sequence.shard_size), and keep pin_memory: true on CUDA.

⸻

RunPod: Run your existing pipeline (no code changes)

Prereqs in RunPod UI
	1.	Create a PyTorch (CUDA 12.x) template pod (Ubuntu).
	2.	Choose GPU (good/value picks): RTX A5000 (24GB) or A100 40GB.
	3.	Add a Persistent Volume (e.g., 50–200 GB) so artifacts and data survive restarts. Mount at /workspace.

  Container shell commands

  # 1) Go to the mounted workspace
cd /workspace

# 2) Clone your repo (SSH or HTTPS)
git clone https://github.com/aamitmakhija/AQP_DL_HybridCNNLSTMTransformer.git
cd AQP_DL_HybridCNNLSTMTransformer

# 3) (Optional) Set your git identity if you’ll push from the pod
git config user.name "Amit Makhija"
git config user.email "amit_makhija@outlook.com"

# 4) Python env
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 5) Install CUDA-enabled PyTorch (matches base image CUDA; adjust cu121 if needed)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 6) Install the rest
# If you have requirements.txt use it; else install the minimal set below.
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install pyarrow pandas numpy pyyaml scikit-learn tqdm
fi

# 7) Put data under raw_data/ (three options)
#   a) RunPod "Upload" → /workspace/AQP_.../raw_data/
#   b) wget/curl from S3/Drive (if you have links)
#   c) rsync/scp from your machine to the pod
mkdir -p raw_data
# Place: airquality.csv, meteorology.csv, station.csv, district.csv, city.csv, weatherforecast.csv

# 8) Run end-to-end
export OMP_NUM_THREADS=8
CONFIG="configs/default.yaml,configs/keep.yaml" bash a_scripts/run_all.sh

Notes
	•	The runner auto-detects CUDA (device=auto → cuda) and uses mixed precision if enabled in configs/default.yaml.
	•	Artifacts land in /workspace/AQP_DL_HybridCNNLSTMTransformer/experiments/artifacts/ (on your persistent volume).
Tar and download if needed:
tar -czf artifacts_$(date +%Y%m%d_%H%M).tar.gz experiments/artifacts

	•	To push code changes back to GitHub from the pod: add your SSH key or use a PAT and git push.

Common tweaks on GPU
	•	Increase batch size (e.g., 512 → 1024 on A100) if you have headroom:
	•	dl.batch_size: 512 → 1024
	•	Prefer amp.dtype: fp16 on CUDA; keep bf16 on A100 if stable.
	•	Bump dl.num_workers to 4–8. Keep pin_memory: true on CUDA.

Optional: Multi-Horizon training
	•	In configs/default.yaml set:

  sequence:
  horizon: [1, 3, 6, 12, 24, 48]
  	•	Re-run run_all.sh. Evaluation will emit per-H metrics and predictions.
