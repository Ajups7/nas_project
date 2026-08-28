# Team Setup Guide

Infra split: a shared always-on CPU VM (Hetzner) holds the code and the ~13GB
of downloaded data; GPU training happens on-demand via RunPod (pay-per-second,
spin up only when actually training). No one needs to store the full dataset
on their own laptop.

## 1. Shared VM (Hetzner)

One person creates the account and provisions:
- A CX-line server (2 vCPU / 4GB RAM is enough for data prep; upgrade if needed)
- A ~200GB volume attached for the data
- Ubuntu 22.04 or later

Add all 3 teammates' SSH public keys to the server (Hetzner console lets you
add multiple keys at creation, or add more later via `~/.ssh/authorized_keys`
on the server).

```bash
# On the VM, one-time setup
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
git clone <your-repo-url> nas-disruptnet
cd nas-disruptnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # created below
```

## 2. Getting the data onto the VM

The data currently lives on the original machine that ran the download
pipelines. Two options:

**A. Re-run the download scripts directly on the VM** (cleanest — the
pipelines are already built and battle-tested):
```bash
python scripts/download_bts_ontime.py
python scripts/download_aspm_airport_analysis.py --reptype r1   # Airport Analysis
python scripts/download_aspm_airport_analysis.py --reptype r5   # EDCT Report
python scripts/download_weather_metar.py
```
Note: the ASPM scripts use Playwright (browser automation) - the VM will
need `playwright install chromium` and its OS dependencies
(`playwright install-deps` on a fresh Ubuntu box) before these will run.

**B. Copy the already-downloaded data directly** (faster, skips re-running
hours of downloads):
```bash
# from the machine that has the data, to the VM
rsync -avz --progress data/ user@your-vm-ip:~/nas-disruptnet/data/
```

## 3. Everyone else: connect via SSH

```bash
ssh youruser@your-vm-ip
cd nas-disruptnet
source .venv/bin/activate
```

Recommend using `tmux` or `screen` so long-running work survives a
disconnect: `tmux new -s work`, and `tmux attach -t work` to resume.

## 4. GPU training (RunPod)

Data prep, feature engineering, and writing/debugging model code happens on
the CPU VM (or locally). For actual training runs:

1. Spin up a Pod on RunPod (recommended starting point: **RTX A5000, 24GB,
   ~$0.27/hr** - plenty for this project's model sizes; step up to RTX 4090
   only if you hit real bottlenecks).
2. Copy the specific data/feature files you need for that run from the VM to
   the Pod (`scp` or `rsync`), or `git pull` your model code onto the Pod.
3. Train, save checkpoints/results back to the VM (or push to git/cloud
   storage) before...
4. **Stop the Pod** as soon as you're done - billing is per-second, so
   there's no reason to leave it running idle.

## 5. Local development (optional)

Each teammate can also develop/debug locally using their own machine's GPU
(if any) for quick iteration before pushing real training runs to RunPod.
Google Colab's free T4 (16GB) is another free option for individual
experimentation, but isn't part of the shared team workflow since it's
tied to a single Google account and has no persistent storage.
