# Team Setup Guide

Infra split: a shared always-on CPU VM (Azure for Students - free, no card
required) holds the ~13GB of downloaded data in one place; GPU training
happens on-demand via RunPod (pay-per-second, spin up only when actually
training). No one needs to store the full dataset on their own laptop, and
each person keeps their own independent git checkout on the VM so branches
never collide.

## 1. Shared VM (Azure for Students)

One person claims the credit and provisions the VM:

1. Go to **azure.microsoft.com/free/students**, verify with a school email
   (or student ID / GitHub Student Pack as fallback) - no credit card needed.
2. Create a VM: **B2s** size (2 vCPU / 4GB RAM - fits Azure for Students'
   3-vCPU free-tier cap), **Ubuntu 22.04 LTS**.
3. Add all 3 teammates' SSH public keys during creation (or add more later
   via `~/.ssh/authorized_keys` on the VM).
4. Attach a data disk sized for the dataset with room to grow (~50GB is
   comfortable for the current ~13GB plus future additions like weather
   HRRR or model checkpoints), mounted at `/srv/nas-data`.

```bash
# On the VM, one-time setup
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
sudo mkdir -p /srv/nas-data
sudo chown $USER:$USER /srv/nas-data   # repeat per teammate, or use a shared group
```

## 2. Getting the data onto the VM (once, shared)

The actual data lives in **one place**, `/srv/nas-data/data/` - not inside
anyone's personal git checkout. Two ways to populate it:

**A. Re-run the download scripts directly on the VM** (cleanest — the
pipelines are already built and battle-tested):
```bash
cd /srv/nas-data
git clone <your-repo-url> _download_tmp && cd _download_tmp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium && playwright install-deps   # for the ASPM scripts
python scripts/download_bts_ontime.py
python scripts/download_aspm_airport_analysis.py --reptype r1   # Airport Analysis
python scripts/download_aspm_airport_analysis.py --reptype r5   # EDCT Report
python scripts/download_weather_metar.py
mv data /srv/nas-data/data
cd .. && rm -rf _download_tmp
```

**B. Copy the already-downloaded data directly** (faster, skips re-running
hours of downloads):
```bash
# from the machine that has the data, to the VM
rsync -avz --progress data/ user@your-vm-ip:/srv/nas-data/data/
```

## 3. Each person: your own checkout, symlinked to the shared data

Don't share one git checkout between all three of you - each person clones
their own copy, on their own branch, so switching branches never affects
anyone else:

```bash
ssh youruser@your-vm-ip
git clone <your-repo-url> ~/work-yourname/nas_project
cd ~/work-yourname/nas_project
ln -s /srv/nas-data/data data          # symlink, not a copy - shared bytes, separate checkout
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git checkout -b your-tier-branch-name
```

`common/data_loader.py` resolves `data/` relative to wherever the repo is
checked out, so the symlink is transparent - no code changes needed, and
the ~13GB only exists once on disk regardless of how many people are
working at once.

Recommend `tmux` or `screen` so long-running work survives a disconnect:
`tmux new -s work`, and `tmux attach -t work` to resume.

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
