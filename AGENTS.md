# ipt_recognition — guide for LLM agents and assistants

This file is written for a language model (Claude Code, Codex, Cursor, Copilot, a
chat assistant fed with repository context) helping someone train, fine-tune or
export an Instrumental Playing Technique (IPT) classifier with this repository.
It states what the code actually does, where the README and the code disagree,
and the traps that cost the most time. Everything here was checked against the
source on the `research` branch. When this file and the code disagree, the code
wins: re-read `preprocess.py`, `train.py`, `finetune.py` and `utils/`.

Humans: the README is the friendlier entry point. This file is denser on purpose.

---

## 1. What this repository is

A PyTorch toolkit that turns folders of audio recordings, one folder per playing
technique, into a **TorchScript classifier (`.ts`)** that runs in real time in
Max/MSP through the `ipt~` external and the `pipo.ipt` module from
[ipt_tilde](https://github.com/nbrochec/ipt_tilde), both built on the
[libipt](https://github.com/nbrochec/libipt) C library. It is part of SPIRIT
(System for Real-Time Recognition of Instrumental Playing Techniques), IRCAM,
REACH project.

The pipeline is three scripts run from the **repository root**:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `preprocess.py` | `data/raw/<split>/<class>/*.wav` | `.pt` segments in `data/preprocessed/<name>/`, dataset CSV + Parquet in `data/dataset/` |
| 2 | `train.py` or `finetune.py` | the dataset CSV/Parquet | `runs/<name>_<date>_<time>/` with `.pth`, `.ts`, optional `.onnx`, `.yaml`; `logs/<same>/` with results and confusion matrix CSVs |
| 3 | (external) `ipt~` / `pipo.ipt` | the `.ts` file | real-time class probabilities in Max |

Every script builds its paths relative to the current working directory
(`data/...`, `runs/...`, `logs/...`, `config/...`). Run them from the repo root or
they will not find anything.

---

## 2. Environment

- Python **3.11.7** in a dedicated conda env (`conda create --name IPT python=3.11.7`).
- `pip install -r requirements.txt`. Versions are pinned: `torch==2.2.2`,
  `torchaudio==2.2.2`, `librosa==0.10.2.post1`, `polars==1.22.0`,
  `torch-audiomentations==0.11.1`, `torchmetrics==1.4.1`, `tensorboardX`,
  `onnx`, `onnxruntime` (the last two unpinned).
- Devices: `--device cpu` or `--device cuda`. There is **no MPS path**;
  `DeviceName.get` raises on anything else. Training on a Mac means CPU.
- `train.py` and `finetune.py` set the multiprocessing start method to `spawn`
  and end with `os._exit(0)`; expect no clean interpreter shutdown.
- The TorchScript produced here is consumed by libtorch **2.4.1** on the Max
  side. It loads fine; do not "upgrade" this repo's torch without checking that
  `torch.jit.script` output still loads in ipt_tilde.

---

## 3. Repository map

```
preprocess.py        split → resample → trim → segment → augment → save .pt, write CSV/Parquet
train.py             train from scratch, export .pth/.ts (+ .onnx)
finetune.py          load a .pth, optionally freeze the CNN, train the head, export
train.ipynb          notebook wrapper for non-experts (same CLI calls)
config/              YAML configs; --config resolves names *inside this folder*
data/raw/            your audio, one sub-folder per class, one folder per split
data/preprocessed/   generated .pt tensors, one per segment (and per augmentation)
data/dataset/        <name>_dataset_split.csv and .parquet (file_path,label_name,label_index,set)
models/models.py     architectures: flute, eguitar, base
models/layers.py     LogMelSpectrogramLayer (+ ONNX variant), AdaptivePool2dONNX, customConv2d
models/models_utils.py  DeviceName, ModelChecker, LoadModel, ModelTrainer, ModelSaver, PrepareModel
utils/dataset_manager.py  DatasetSplitter, DatasetMaker, ConfigFile, SegLenConverter, CSVDataset, …
utils/audio.py       AudioPreprocessor, offline + online augmentations, AttackAugmenter
utils/prepare_data.py  Datasets + balanced batch sampler → DataLoaders
utils/save_results.py  TensorBoard and CSV reporting
externals/pytorch_balanced_sampler/  vendored class-balanced sampler (Karl Hornlund)
pretrained/          flute_pretrained.pth (state_dict for the `base` arch), *_help.ts demo models
.github/workflows/   ci-research.yml, ci-performance.yml (smoke runs on synthetic data)
```

Only `.gitkeep` files are tracked under `data/`, and `runs/` and `logs/` are
untracked. `.gitignore` lists only `__pycache__/` and `.DS_Store`, so a careless
`git add -A` **would commit datasets and checkpoints**. Add paths explicitly.

---

## 4. Branches

| Branch | Role |
|---|---|
| `research` | this file's branch; CI badge in README points here |
| `performance` | remote default (`origin/HEAD`); diverged from `research` after `a69758c` with a Google Colab notebook, quieter console output and a `num_classes` fix |
| `dev` | ancestor of `research` (research = dev + the CI split commits) |

Before changing shared code, ask which branch the user is working on. A fix
present on one branch may be missing on another.

---

## 5. Step 1 — preprocessing (`preprocess.py`)

```bash
python preprocess.py --name my_run --sampling_rate 24000 --segment_length "1000 ms" --val_split train
```

What it does, in order:

1. **Validation split.** If `--val_dir` is not given, `DatasetSplitter` picks
   a stratified 20 % (`--val_ratio`) of the files from `--val_split` (`train` or
   `test`) and **moves them** (`shutil.move`) into `data/raw/val_<name>/`. The raw
   training folder is modified on disk. Keep a copy of raw data elsewhere. If
   `--val_dir` is given, nothing is moved.
2. **Segment length** is parsed from the string `"<int> samps"` or `"<int> ms"`
   and converted to samples at `--sampling_rate`. Default `"14700 samps"`
   (333 ms at 44.1 kHz).
3. **Per file:** load with librosa, resample to `--sampling_rate`, trim leading
   and trailing silence (`librosa.effects.trim`), pad to one segment if shorter
   (`--padding minimal`, default) or pad the last partial segment
   (`--padding full`), cut into consecutive segments (`--segment_overlap` gives
   a half-segment hop). Accepted extensions: `.wav .mp3 .flac .aiff .aif`.
   Spaces in file names are replaced by underscores when listing files.
4. **Offline augmentation, training split only, always on:** each segment is
   saved four times: `original`, `detuned` (random ±100 cents around A4),
   `noise` (Gaussian, σ = 0.01), `stretched` (rate 0.9–1.1). Test and val
   segments are saved once. There is **no `--offline_augment` flag** despite the
   README table, and `--use_original` is parsed but never read: the original is
   always saved.
5. `--attack 1` additionally creates three copies of every `segment0` file with
   25 / 50 / 75 % leading silence (`_shifted{0,1,2}.pt`).
6. `DatasetMaker` scans `data/preprocessed/<name>/{train,test,val}` and writes
   `data/dataset/<name>_dataset_split.csv` (+ `.parquet`) with columns
   `file_path,label_name,label_index,set`. The **label is the name of the folder
   that directly contains the audio file**, and `label_index` is assigned in
   **sorted order of the training class names**. Nested "bank" folders
   (`train/bank_A/class_1/…`) are supported: the leaf folder is the class.
7. `DatasetValidator` requires train, test and val to have the **same set of
   class names**, otherwise it raises.

Gotchas:

- `--val_split` has **no default in the code** (the README says `train`). With
  neither `--val_dir` nor `--val_split`, the script raises
  `val_split must be 'train' or 'test'`. Always pass one of them.
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
- The checkpoint is written **every time validation loss improves**; the best
  state is reloaded before testing. `--early_stopping N` stops after N epochs
  without improvement.
- Metrics: micro accuracy, macro F1, loss on the test split. Reported to
  TensorBoard (`runs/<run>/`) and to `logs/<run>/results_<run>.csv` and
  `cm_<run>.csv`.
- Exports: `<run>.pth` (state_dict), `<run>.ts` (`torch.jit.script` of the
  model moved to CPU), and `<run>.onnx` if `--onnx 1` (opset 17, dynamic batch
  and sample axes, metadata `sr`, `seglen`, `classnames` embedded).
- `--onnx 1` also **changes the model at construction**: the log-mel layer is
  swapped for a manual-STFT ONNX-compatible implementation. Numerics differ
  slightly from the torchaudio path, so compare against a non-ONNX run before
  trusting metrics.

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
python preprocess.py --name my_ft --sampling_rate 44100 --segment_length "14700 samps" --val_split train
python finetune.py  --name my_ft --pretrained pretrained/flute_pretrained.pth --model base \
                    --n_mels 384 --hop_length 128 --sampling_rate 44100 --segment_length "14700 samps"
```

- Defaults are set for the shipped checkpoint: `--model base`, `--n_mels 384`,
  `--hop_length 128`, `--sampling_rate 44100`, `--segment_length "14700 samps"`,
  `--learning_rate 0.0003`, `--epochs 50`, `--early_stopping 10`, `--reduce_lr 1`.
- Weights are copied layer by layer where **names and shapes match**; the
  final `fc` layer with a different class count keeps its random init.
- `verify_audio_parameters` raises on a sample-rate or segment-length mismatch
  with the checkpoint, and only warns on mel parameter differences.
- `--freeze_conv 1` (default) trains only parameters whose name contains `fc`
  or `logmel`.
- Runs are named `<name>_finetune_<date>_<time>`. No ONNX export here.

**Known defect on this branch:** `finetune.py` builds
`ModelTrainer(model, loss_fn, args.device, transform)` while `ModelTrainer.__init__`
requires a fifth argument `num_classes`. As written this raises a `TypeError`
before the first epoch. `TestSavedModel.test_model` in `models/models_utils.py`
has the same three-argument call. CI does not catch it because every training
step runs with `|| true`. If asked to fine-tune, check `git log`/`git diff`
first: the fix is a one-line change passing `args.num_classes`.

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

Two identical-shaped workflows, `ci-research.yml` (on `research`) and
`ci-performance.yml` (on `performance`), on `ubuntu-latest`: install the pinned
requirements, synthesise a 2-class dataset (440 Hz sine vs white noise, 1 s
clips), preprocess, train `flute` and `eguitar` for one epoch with `--onnx 1`,
train from `config_example.yaml`, then preprocess and fine-tune `base` from the
shipped checkpoint. Almost every step ends in `|| true`, so the workflow is a
**smoke run, not a gate**: a green badge means the scripts ran, not that they
succeeded. Read the job log when in doubt.

---

## 10. How to work in this repo as an agent

- Reproduce a user's problem with the CI recipe: it needs no real data and
  finishes in minutes on CPU.
- Change `argparse` definitions and the README tables together; they have
  already drifted (see §5).
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

## 11. Related repositories

- [ipt_tilde](https://github.com/nbrochec/ipt_tilde) — the Max package
  (`ipt~`, `pipo.ipt`) that runs the exported `.ts`. It has its own
  `AGENTS.md`.
- [libipt](https://github.com/nbrochec/libipt) — the C library both externals
  use; `include/ipt.h` is the public contract, `ARCHITECTURE.md` explains the
  internals.
- Papers: Brochec et al. 2024/2025/2026, Fiorini et al. 2025 (see README).
