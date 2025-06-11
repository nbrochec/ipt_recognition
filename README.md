# 🎵 Real-Time Instrumental Playing Techniques Recognition

This repository provides a toolkit for the real-time recognition of instrumental playing techniques using deep learning CNN models. It allows the user to train his own classification model using his personal data. The toolkit also provide pre-trained models for flute and eguitar IPT recognition.

## 🚀 Installation
Clone this repository, navigate to the folder, create a conda environment with Python 3.11.7, and install requirements.
```bash
git clone https://github.com/nbrochec/realtimeIPTrecognition/
cd realtimeIPTrecognition
conda create --name IPT python=3.11.7
conda activate IPT
pip install -r requirements.txt
```

Non-expert users: we recommend you to use our Jupyter Notebook `train.ipynb`

## 📁 Main Folders
```
└── 📁config            # Configuration files
└── 📁data              # Dataset and preprocessed data
    └── 📁dataset       # Dataset CSV files
    └── 📁preprocessed  # Preprocessed audio files
    └── 📁raw           # Raw audio files
        └── 📁test      # Test dataset
        └── 📁train     # Training dataset
        └── 📁val       # Validation dataset (optional)
└── 📁models            # Model architectures
└── 📁pre-trained       # Pre-trained models
└── 📁utils             # Utility functions
```

## 📦 Usage
### 📂 Dataset Preparation
Place your training audio files in `/data/raw/train/` and test files in `/data/raw/test/` (val files is also possible in `/data/raw/val/`). Each IPT class should have its own folder with the same name in both train, test and val directories.

Example structure:
```
└── 📁train
    └── 📁IPTclass_1
        └── audiofile1.wav
        └── audiofile2.wav
    └── 📁IPTclass_2
        └── audiofile1.wav
        └── audiofile2.wav
```

You can use multiple training datasets. They must share the same names for IPT classes.

### 🔄 Preprocess your datasets
Use `screen` to access multiple separate login session insde a single terminal window.
Open a screen and navigate to the root folder.
```bash
screen -S IPT
conda activate IPT
cd realtimeIPTrecognition
```

Preprocess your datasets:
```bash
python preprocess.py --name your_project_name
```

| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|
| `-n`, `--name`      | Name of the project.                                                | String                         |          |
| `-sr`, `--sampling_rate` | Sampling rate for downsampling the audio files.                     | Integer (Hz) > `0`             | `44100`         |
| `--train_dir` | Directory of training samples to preprocess.                        | String                       | `train`         |
| `--test_dir`  | Directory of test samples to preprocess.                           | String                        | `test`          |
| `--val_dir`   | Directory of validation samples to preprocess.                    | String                         | `None`          |
| `--val_split` | Specify on which dataset the validation split would be made.      | `train`, `test`                | `train`         |
| `--val_ratio` | Amount of validation samples.                                       | `0` < Float value < `1`         | `0.2`           |
| `--offline_augment` | Use offline augmentations to generate data using detuning, gaussian noise and time stretching from original audio files. | `0` or `1` | `1` |
| `--use_original`       | Use original data as training data.                             | `0` or `1`               | `1`        |
| `--padding`         | Pad the arrays of audio samples with zeros. `minimal` only pads when audio file length is shorter than required input length. `full` pads any segment that has less than the required input length. | `full`, `minimal`, `None`  | `minimal`           |


Notes:
- If `--val_dir` is not specified, the validation set will be generated from the folder specified with `--val_split`
- Downsampling audio files is recommended, use `--sampling_rate` argument for that purpose
- Preprocessed audio files will be saved to `/data/preprocessed` folder
- A CSV file will be saved in the `/data/dataset/` folder with the following syntax: `your_project_name_dataset_split.csv`

### 🏋️ Training
There are many different configurations for training your model. The only required argument is `--name`. Use the same name as well as the same sampling rate as used with `preprocessing.py`.
```bash
python train.py --name your_project_name
```

You can use the following arguments if you want to test different configurations.
| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|  
| `-n`, `--name`      | Name of the project.                                              | String                         |           |
| `-d`, `--device`    | Specify the hardware on which computation should be performed.     | `cpu`, `cuda`           | `cpu`           |
| `-m`, `--model`     | Name of the model's architecture.                                  | `eguitar`, `flute`  | `flute`   |
| `-sr`, `--sampling_rate` | Specify sampling rate. The use of the same sampling rate as in preprocess.py is recommended.  | Integer (Hz)     | `44100`         |
| `--segment_overlap` | Overlap between audio segments. Increase the data samples by a factor 2. | `0` or `1`                |   `0`       |
| `--segment_length`   | Defines segment length of audio data samples. You should precise either if it is samples `samps` or millisecond `ms`  | String |  `"14700 samps"`      |
| `--f_min`            | Minimum frequency for Mel filters.                                 | Integer (Hz)                    | `20`          |
| `--f_max`            | Maximum frequency for Mel filters.                                 | Integer (Hz)                    |           |
| `--n_mels`           | Number of Mel filters.                                              | Integer > `0`                   | `128`         |
| `--n_fft`            | Size of the FFT window.                                             | Integer > `0`                    | `2048`      |
| `--hop_length`       | Hop length for downsampling the audio files.                        | Integer > `0`                  | `512`         |
| `-lr`,`--learning_rate`    | Learning rate.                                               | Float value > `0`              | `0.001`         |
| `--batch_size`       | Specify batch size.                                               | Integer value > `0`            | `128`         |
| `--epochs`          | Number of training epochs.                                         | Integer value > `0`            | `100`           |
| `--online_augment`  | Use online augmentations based on polarity inversion, lpf and hpf. | `0` or `1` |      `0`     |
| `--num_workers`     | Number of workers for data loading.                                | Integer value > `0`            | `4`            |
| `--early_stopping`  | Number of epochs without improvement before early stopping.         | Integer value > `0`   |        |
| `--reduce_lr`        | Reduce learning rate if validation plateaus.                       | `0` or `1`             |     `0`    |

You can also pass a `.yaml` configuration file using `--cfg` argument:
```bash
python train.py --name your_project_name --cfg your_config_file.yaml
```
### 📊 Monitoring Training
1. Detach from current screen: `ctrl`+`A`+`D`
2. Open a new screen:
```bash
screen -S monitor
conda activate IPT
cd realtimeIPTrecognition
```
3. Start tensorboard:
```bash
tensorboard --logdir . --bind_all
```
4. If working on a remote server, connect to tensorboard:
```bash
ssh -L 6006:localhost:6006 user@server
```
### 📁 Output Files
After training, the following files will be created:

In `/runs/your_project_name_date_time/`:
- Model checkpoints (`.pth`)
- Torchscript model (`.ts`)
- Configuration (`.yaml`)

In `/logs/your_project_name_date_time/`:
- Confusion matrix (`.csv`)
- Results (`.csv`)

## 💻 Running the model in real-time using Max/MSP

We developed a dedicated Max/MSP object to run the exported `.ts` model in real-time.
Follow the instructions of the [ipt_tilde](https://github.com/nbrochec/ipt_tilde) repository.

## 🧠 About
This project is part of an ongoing research effort into the real-time recognition of instrumental playing techniques for interactive music systems.

### 📚 Citations
- [Brochec et al. (2025)](https://hal.science/hal-05061669) - "Interactive Music Co-Creation with an Instrumental Technique-Aware System: A Case Study with Flute and Somax2"
- [Fiorini et al. (2025)](https://hal.science/hal-05061680) - "Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques"
- [Brochec et al. (2024)](https://hal.science/hal-04642673) - "Microphone-based Data Augmentation for Automatic Recognition of Instrumental Playing Techniques"

### 🙏 Acknowledgments
This project uses code from the [pytorch_balanced_sampler](https://github.com/khornlund/pytorch-balanced-sampler) repository created by Karl Hornlund.

## 📜 License and Fundings

This project is released under a CC-BY-NC-4.0 license.

This research is supported by the European Research Council (ERC) as part of the [Raising Co-creativity in Cyber-Human Musicianship (REACH) Project](https://reach.ircam.fr) directed by Gérard Assayag, under the European Union's Horizon 2020 research and innovation program (GA \#883313). 
Funding support for this work was provided by a Japanese Ministry of Education, Culture, Sports, Science and Technology (MEXT) scholarship to Nicolas Brochec. 
