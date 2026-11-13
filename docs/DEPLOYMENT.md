# Deployment / 部署

## Environments

Recommended: 64-bit Python 3.12, PyTorch 2.7.1, NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0, Streamlit 1.45.1, PyYAML 6.0.2. Fixed application versions are in `requirements.txt`.

Original training: Python 3.12.3, PyTorch 2.7.0a0+7c8ec84dab.nv25.3, CUDA 12.8, NumPy 1.26.4, RTX 4090. These are separate environment descriptions. CPU inference needs no CUDA.

## Windows PowerShell

```powershell
git clone https://github.com/cuitao223/LanWind.git
cd LanWind
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

The page starts without data and displays model information. Inference requires your own file following `DATA.md`.

## Linux CPU

```bash
git clone https://github.com/cuitao223/LanWind.git
cd LanWind
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/infer.py --input /private/input.npz --output runs/prediction.npz
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

## CUDA server

Use a separate environment, preserving unrelated data and processes. Install a compatible driver before selecting the CUDA wheel.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
mkdir -p runs
nohup python -u scripts/train.py --data /private/bundles --output runs/my_run --device cuda --batch-size 32 > runs/my_run.log 2>&1 &
echo $!
```

Monitor the PID, `nvidia-smi`, log and `training.csv`. Batch 32 matches the original run; reduce if available memory requires it. The trainer never stops unrelated jobs and refuses CPU fallback. It saves normalization, configuration, partition audit, CSV curves and best validation weights; it never reads test bundles.

## Remote visualization

```bash
nohup python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 > runs/dashboard.log 2>&1 &
```

On your machine, use your own SSH alias:

```bash
ssh -N -L 8502:127.0.0.1:8502 YOUR_SERVER_ALIAS
```

Open `http://127.0.0.1:8502`. File paths refer to the machine running Streamlit. No server addresses, SSH credentials or datasets are included.
