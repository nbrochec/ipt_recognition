# ipt_recognition — guide for LLM agents and assistants

This file is written for a language model (Claude Code, Codex, Cursor, Copilot, a
chat assistant fed with repository context) helping someone train, fine-tune or
export an Instrumental Playing Technique (IPT) classifier with this repository.
It states what the code actually does, where the README and the code disagree,
and the traps that cost the most time. Everything here was checked against the
source on the `performance` branch (the default branch). The `research` branch
has its own `AGENTS.md`, and the two pipelines differ (see §4). When this file
and the code disagree, the code wins: re-read `preprocess.py`, `train.py`,
`finetune.py` and `utils/`.

Humans: the README is the friendlier entry point. This file is denser on purpose.

---

## 1. What this repository is

A PyTorch toolkit that turns folders of audio recordings, one folder per playing
technique, into a **TorchScript classifier (`.ts`)** that runs in real time in
Max/MSP through the `ipt~` external and the `pipo.ipt` module from
[ipt_tilde](https://github.com/nbrochec/ipt_tilde), both built on the
[libipt](https://github.com/nbrochec/libipt) C library. It is part of SPIRIT
(System for Real-Time Recognition of Instrumental Playing Techniques), IRCAM,
REACH project. The repository lives at
[DYCI2/ipt_recognition](https://github.com/DYCI2/ipt_recognition).

The pipeline is three scripts run from the **repository root**:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `preprocess.py` | `data/raw/<train_dir>/<class>/*.wav` | copies in `data/raw/{train,val}_<name>/`, `.pt` segments in `data/preprocessed/<name>/`, dataset CSV + Parquet in `data/dataset/` |
| 2 | `train.py` or `finetune.py` | the dataset CSV/Parquet | `runs/<name>_<date>_<time>/` with `.pth`, `.ts`, optional `.onnx` (`train.py` only), `.yaml`, TensorBoard events |
| 3 | (external) `ipt~` / `pipo.ipt` | the `.ts` file | real-time class probabilities in Max |

Every script builds its paths relative to the current working directory
(`data/...`, `runs/...`, `config/...`). Run them from the repo root or they will
not find anything.

---

## 2. Environment

- Python **3.11.7** in a dedicated conda env (`conda create --name IPT python=3.11.7`).
  Any 3.11.x works; the pinned `torch==2.2.2`, `numba==0.60.0` and
  `llvmlite==0.43.0` have wheels only up to Python **3.12**, so 3.13 fails at
  `pip install` with "failed to build numba".
- `pip install -r requirements.txt`. Versions are pinned: `torch==2.2.2`,
  `torchaudio==2.2.2`, `librosa==0.10.2.post1`, `polars==1.22.0`,
  `torch-audiomentations==0.11.1`, `torchmetrics==1.4.1`, `tensorboardX`,
  `onnx`, `onnxruntime` (the last two unpinned).
- Devices: `--device cpu` or `--device cuda`. There is **no MPS path**;
  `DeviceName.get` raises on anything else. Training on a Mac means CPU.
- `train.py` and `finetune.py` set the multiprocessing start method to `spawn`,
  silence all Python warnings (`warnings.filterwarnings("ignore")`) and end with
  `os._exit(0)`; expect no clean interpreter shutdown.
- The TorchScript produced here is consumed by libtorch **2.4.1** on the Max
  side. It loads fine; do not "upgrade" this repo's torch without checking that
  `torch.jit.script` output still loads in ipt_tilde.

---

## 3. Repository map

```
preprocess.py        split → resample → trim → segment → augment → save .pt, write CSV/Parquet
train.py             train from scratch, export .pth/.ts (+ .onnx)
finetune.py          load a .pth, optionally freeze the CNN, train the head, export .pth/.ts
train.ipynb          notebook wrapper for non-experts (same CLI calls)
train_colab.ipynb    Google Colab version of the notebook (see §10)
config/              YAML configs; --config resolves names *inside this folder*
data/raw/            your audio, one sub-folder per class
data/preprocessed/   generated .pt tensors, one per segment (and per augmentation)
data/dataset/        <name>_dataset_split.csv and .parquet (file_path,label_name,label_index,set)
models/models.py     architectures: flute, eguitar, base
models/layers.py     LogMelSpectrogramLayer (+ ONNX variant), AdaptivePool2dONNX, customConv2d
models/models_utils.py  DeviceName, ModelChecker, LoadModel, ModelTrainer, ModelSaver, PrepareModel
utils/dataset_manager.py  DatasetSplitter, DatasetMaker, ConfigFile, SegLenConverter, CSVDataset, …
utils/audio.py       AudioPreprocessor, offline + online augmentations, AttackAugmenter
utils/prepare_data.py  Datasets + balanced batch sampler → train and val DataLoaders
utils/save_results.py  TensorBoard and CSV reporting (CSV reporting unused on this branch)
externals/pytorch_balanced_sampler/  vendored class-balanced sampler (Karl Hornlund)
pretrained/          flute_pretrained.pth (state_dict for the `base` arch), *_help.ts demo models
.github/workflows/   ci-performance.yml (active), run-commands.yml (legacy, triggers on `main` only)
```

Only `.gitkeep` files are tracked under `data/`, and `runs/` and `logs/` are
untracked. `.gitignore` lists only `__pycache__/` and `.DS_Store`, so a careless
`git add -A` **would commit datasets and checkpoints**. Add paths explicitly.

---

## 4. Branches

| Branch | Role |
|---|---|
| `performance` | this file's branch; **default branch** (`origin/HEAD`), README badges (CI, Colab) point here. Simplified pipeline: one input folder split into train/val, no test set, quieter console output, `num_classes` fix in `finetune.py`, Google Colab notebook |
| `research` | extended pipeline: separate train/test/val folders, `--val_dir`/`--val_split`, test-set metrics and confusion matrix written to `logs/`. Diverged from `performance` after `a69758c` |
| `dev` | ancestor of `research` |

Before changing shared code, ask which branch the user is working on. A fix
present on one branch may be missing on another; for example `research` still
has the `finetune.py` `num_classes` defect fixed here.

---

## 5. Step 1 — preprocessing (`preprocess.py`)

```bash
python preprocess.py --name my_run --train_dir my_data_folder --sampling_rate 24000 --segment_length "1000 ms"
```

What it does, in order:

1. **Train/val split.** `DatasetSplitter` collects every audio file under
   `data/raw/<train_dir>/` (default `train`), splits them stratified by class
   with `--val_ratio` (default 0.2, `random_state=42`) and **copies**
   (`shutil.copy2`) them into `data/raw/train_<name>/` and `data/raw/val_<name>/`.
   The source folder is left untouched, but the data takes twice the disk
   space. `args.train_dir` and `args.val_dir` are then rewritten to those two
   folder names for the rest of the run. There is no `--val_dir`, `--val_split`
   or `--test_dir` on this branch.
2. **Segment length** is parsed from the string `"<int> samps"` or `"<int> ms"`
   and converted to samples at `--sampling_rate`. Default `"14700 samps"`
   (333 ms at 44.1 kHz).
3. **Per file:** load with librosa, resample to `--sampling_rate`, trim leading
   and trailing silence (`librosa.effects.trim`), pad to one segment if shorter
   (`--padding minimal`, default) or pad the last partial segment
   (`--padding full`), cut into consecutive segments (`--segment_overlap` gives
   a half-segment hop). Accepted extensions: `.wav .mp3 .flac .aiff .aif`.
   Spaces in file names are replaced by underscores when listing files.
4. **Offline augmentation, training split only,** controlled by
   `--offline_augment` (default 1): each training segment is then saved four
   times: `original`, `detuned` (random ±100 cents around A4), `noise`
   (Gaussian, σ = 0.01), `stretched` (rate 0.9–1.1). With `--offline_augment 0`,
   or for val segments, only `original` is saved. `--use_original` is parsed
   but never read: the original is always saved.
5. `--attack 1` additionally creates three copies of every `segment0` file with
   25 / 50 / 75 % leading silence (`_shifted{0,1,2}.pt`).
6. `DatasetMaker` scans `data/preprocessed/<name>/{train_<name>,val_<name>}` and
   writes `data/dataset/<name>_dataset_split.csv` (+ `.parquet`) with columns
   `file_path,label_name,label_index,set` (`set` is `train` or `val`). The
   **label is the name of the folder that directly contains the audio file**,
   and `label_index` is assigned in **sorted order of the training class
   names**. Nested "bank" folders (`my_data/bank_A/class_1/…`) are supported:
   the leaf folder is the class.
7. `DatasetValidator` requires train and val to have the **same set of class
   names**, otherwise it raises. A class with a single audio file already
   makes the stratified split in step 1 raise (`The least populated classes in
   y have only 1 member`); give every class at least a few files.

Gotchas:

- **Re-running with the same `--name` does not clean up.** The copies are
  written into existing `train_<name>/` and `val_<name>/` folders. If the source
  folder changed in between (files added, removed or renamed), the new split
  differs, stale copies stay behind, and a file can end up in **both** train
  and val, which inflates validation metrics. Use a new `--name`, or delete
  `data/raw/{train,val}_<name>/` and `data/preprocessed/<name>/` first (ask the
  user before deleting anything under `data/raw/`).
- Preprocessing is parallel (`ProcessPoolExecutor`) and swallows per-file
  exceptions with a printed `Error processing file …`. Read the log; a silent
  class with zero files will fail later in the validator or the sampler.
- Segments are stored as float32 1-D tensors of exactly `segment_length`
  samples. Any later step assumes that length.

---

## 6. Step 2 — training (`train.py`)

```bash
python train.py --name my_run --sampling_rate 24000 --segment_length "1000 ms" \
                --model flute --device cpu --epochs 100 --online_augment 1 --early_stopping 10
```

- `--name` must match the preprocessing name (it selects
  `data/dataset/<name>_dataset_split.csv`). `--sampling_rate` and
  `--segment_length` must match what was used in preprocessing: the model bakes
  the sample rate and segment length into its metadata, and libipt trusts them.
- `--model` is one of `flute` (default), `eguitar`, `base`. The names come from
  the papers; any architecture can be trained on any instrument.
- Batches come from a **class-balanced batch sampler** (`alpha=1`, `kind='fixed'`),
  so per-class imbalance in the training set is compensated at batch level.
- `--online_augment 1` applies, on the fly, polarity inversion, a high-pass
  filter between `f_min` and `2·f_min`, and a low-pass filter between `sr/4` and
  `sr/2 − 100`, each with probability 0.5.
- Optimizer Adam, `weight_decay=1e-5`, `CrossEntropyLoss`.
  `--reduce_lr 1` adds `ReduceLROnPlateau(patience=10, factor=0.1)`.
- `--early_stopping` defaults to **5** here: training stops after 5 epochs
  without validation-loss improvement unless told otherwise.
- The checkpoint is written **every time validation loss improves**; the best
  state is reloaded before export.
- **There is no test set and no final evaluation.** The only metrics are the
  per-epoch validation loss, accuracy and F1 sent to TensorBoard
  (`runs/<run>/`). They are optimistic, because the checkpoint is selected on
  that same validation set. Nothing is written to `logs/`: the test and
  `ToDisk` calls are commented out. For a held-out evaluation, use `research`.
- Exports: `<run>.pth` (state_dict), `<run>.ts` (`torch.jit.script` of the
  model moved to CPU), `<run>.yaml`, and `<run>.onnx` if `--onnx 1` (opset 17,
  dynamic batch and sample axes, metadata `sr`, `seglen`, `classnames`
  embedded).
- `--onnx 1` also **changes the model at construction**: the log-mel layer is
  swapped for a manual-STFT ONNX-compatible implementation. Numerics differ
  slightly from the torchaudio path, so compare against a non-ONNX run before
  trusting metrics.
- The README training table lists `--segment_overlap`, which is a
  `preprocess.py` option; `train.py` does not accept it.

### YAML configs (`--config`)

`--config my.yaml` is resolved as `config/my.yaml` relative to the repo, not as
an arbitrary path. Keys are the human-readable ones written by `ConfigFile.save`
(see `config/config_example.yaml`), mapped to argparse names by lower-casing and
replacing spaces with underscores: `Sampling Rate` → `sampling_rate`,
`N FFT` → `n_fft`, `F min` → `f_min`, `Segment Length` → `segment_length`
(string with unit, converted after load). **Every key in the YAML overrides the
CLI**, including `Name`. Each run also writes its own YAML into `runs/<run>/`,
which can be replayed with `--config` after copying it into `config/`.

---

## 7. Fine-tuning (`finetune.py`)

```bash
python preprocess.py --name my_ft --train_dir my_ft_data --sampling_rate 44100 --segment_length "14700 samps"
python finetune.py  --name my_ft --pretrained pretrained/flute_pretrained.pth --model base \
                    --n_mels 384 --hop_length 128 --sampling_rate 44100 --segment_length "14700 samps"
```

- Defaults are set for the shipped checkpoint: `--model base`, `--n_mels 384`,
  `--hop_length 128`, `--sampling_rate 44100`, `--segment_length "14700 samps"`,
  `--learning_rate 0.0003`, `--epochs 50`, `--early_stopping 5`, `--reduce_lr 1`
  (`ReduceLROnPlateau(patience=5, factor=0.5)`).
- Weights are copied layer by layer where **names and shapes match**; the
  final `fc` layer with a different class count keeps its random init.
- `verify_audio_parameters` raises on a sample-rate or segment-length mismatch
  with the checkpoint, and only warns on mel parameter differences.
- `--freeze_conv 1` (default) trains only parameters whose name contains `fc`
  or `logmel`.
- Runs are named `<name>_finetune_<date>_<time>`. Exports `.pth`, `.ts` and the
  YAML; no ONNX export here. Like `train.py`, there is no test evaluation.
- `ModelTrainer` is built with `args.num_classes` here, so fine-tuning runs.
  `TestSavedModel.test_model` in `models/models_utils.py` still calls
  `ModelTrainer(model, loss_fn, device)` with three arguments and would raise
  a `TypeError`; nothing on this branch calls it.

---

## 8. The model contract with ipt~ / libipt

Anything exported from here must satisfy what libipt loads
(`libipt/core/model.h` in the ipt_tilde tree):

| Requirement | Where it comes from |
|---|---|
| `forward(x)` with `x` of shape `(batch, 1, segment_length)`, float32 raw audio, returns logits `(batch, num_classes)` | all three architectures |
| exported methods `get_sr() -> int`, `get_seglen() -> int`, `get_classnames() -> List[str]` | `@torch.jit.export` on each model |
| softmax is applied **by libipt**, not by the model | keep returning logits |
| class order = `sorted(label names)` | `CSVDataset.get_names_and_nbr` |

libipt resamples the incoming audio to `get_sr()`, accumulates
`get_seglen()` samples, and calls `forward` once per Max signal block, so the
segment length is also the classification latency floor. The log-mel front end
is inside the model; the Max side sends raw audio only.

Architecture summary:

| Name | Front end | Notes |
|---|---|---|
| `flute` | three log-mel layers, `n_fft` 512/1024/2048, stacked as 3 channels, adaptive-pooled to 384×112 | default; pooling makes it tolerant to `n_mels`/`hop_length` changes |
| `eguitar` | one log-mel layer, `n_fft` 2048, adaptive-pooled to 128×15, deeper CNN (512 ch) | larger model |
| `base` | three log-mel layers like `flute`, **no adaptive pool** | the arch of `pretrained/flute_pretrained.pth`; input-size sensitive, so keep `n_mels`/`hop_length`/`seglen` identical to the checkpoint |

All convolutions are `customConv2d` = Conv2d + BatchNorm2d + LeakyReLU(0.01),
weights Xavier-normal initialised. The log-mel output is min-max normalised to
[0, 1] per example.

---

## 9. CI

`ci-performance.yml` runs on pushes and pull requests to `performance`, on
`ubuntu-latest` with `python-version: '3.11'` (latest 3.11.x; 3.11.7 itself has
no build for the Ubuntu 26.04 runners): install the pinned requirements,
synthesise a 2-class dataset (440 Hz sine vs white noise, 1 s clips),
preprocess, train `flute` and `eguitar` for one epoch with `--onnx 1`, train
from `config_example.yaml`, verify the ONNX export, then preprocess and
fine-tune `base` from the shipped checkpoint. Almost every step ends in
`|| true`, so the workflow is a **smoke run, not a gate**: a green badge means
the scripts ran, not that they succeeded. Read the job log when in doubt.

`run-commands.yml` is an older copy that only triggers on a `main` branch,
which does not exist; it never runs.

---

## 10. Google Colab (`train_colab.ipynb`)

Opened from the README badge
(`colab.research.google.com/github/DYCI2/ipt_recognition/blob/performance/train_colab.ipynb`).
It clones this branch, copies the user's class folders from Google Drive into
`data/raw/`, then runs `preprocess.py`, `train.py` and optionally `finetune.py`,
and copies the run folder back to Drive.

Colab's default runtime is **Python 3.13** (runtime 2026.10), where the pinned
requirements cannot install. The notebook therefore:

- creates a separate Python 3.11 environment with `uv`
  (`uv venv --python 3.11 /content/ipt_env`) and installs `requirements.txt`
  into it; Colab's own Python and torch are only used for the GPU check cell;
- runs every script as `!MPLBACKEND=Agg /content/ipt_env/bin/python <script>.py …`.
  `MPLBACKEND=Agg` is required: Colab exports
  `MPLBACKEND=module://matplotlib_inline.backend_inline`, which subprocesses
  inherit, and that module does not exist in `/content/ipt_env`, so
  `import seaborn` (via `utils/save_results.py`) crashes.

When editing the notebook, keep both points, and keep `torch==2.2.2` rather than
using Colab's preinstalled torch, so the exported `.ts` still loads in
libtorch 2.4.1. The notebook JSON is written with `indent=1`; edit it with a
script that round-trips `json.load`/`json.dumps(indent=1, ensure_ascii=False)`
to keep diffs small.

---

## 11. How to work in this repo as an agent

- Reproduce a user's problem with the CI recipe: it needs no real data and
  finishes in minutes on CPU.
- Change `argparse` definitions and the README tables together; they have
  already drifted (see §6).
- Keep `preprocess.py`, `train.py` and `finetune.py` argument names aligned
  with each other; users pass the same values to all three.
- Do not touch `externals/pytorch_balanced_sampler/` (vendored, MIT, credited).
- Never move or delete anything under `data/raw/` beyond what
  `DatasetSplitter` already does; users keep their only copy of recordings
  there more often than they should.
- Stage files explicitly; see the note on `.gitignore` in §3.
- The maintainer makes commits and releases. Prepare changes, explain them,
  and hand back the exact `git` commands unless told otherwise.
- License is CC-BY-NC-4.0: non-commercial. Say so if asked about reuse.

---

## 12. Related repositories

- [ipt_tilde](https://github.com/nbrochec/ipt_tilde) — the Max package
  (`ipt~`, `pipo.ipt`) that runs the exported `.ts`. It has its own
  `AGENTS.md`.
- [libipt](https://github.com/nbrochec/libipt) — the C library both externals
  use; `include/ipt.h` is the public contract, `ARCHITECTURE.md` explains the
  internals.
- Papers: Brochec et al. 2024/2025/2026, Fiorini et al. 2025 (see README).
