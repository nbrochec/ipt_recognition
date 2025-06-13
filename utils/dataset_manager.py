import os
import csv
import yaml
import shutil
import polars as pl
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split

class RunDirectory:
    @staticmethod
    def get(name):
        """Create runs and the current run directories."""
        cwd = os.getcwd()
        runs = os.path.join(cwd, 'runs')
        if not os.path.exists(runs):
            os.makedirs(runs, exist_ok=True)
        current_run = os.path.join(runs, name)
        return current_run


class ConfigFile:
    """Save Hyperparameters to disk as YAML file."""
    def __init__(self, args):
        self.args = args

    def save(self, current_run):
        cwd = os.getcwd()
        path_to_run = current_run

        if not os.path.exists(os.path.join(cwd, 'runs')):
            os.makedirs(os.path.join(cwd, 'runs'))

        if not os.path.exists(path_to_run):
            os.makedirs(path_to_run)

        current_config = {
            'Name': self.args.name,
            'Device': self.args.device,
            'Model': self.args.model,
            'Sampling Rate': self.args.sampling_rate,
            'Segment Overlap': self.args.segment_overlap,
            'F min': self.args.f_min,
            'F max': self.args.f_max,
            'Offline Augment': self.args.offline_augment,
            'Online Augment': self.args.online_augment,
            'Learning Rate': self.args.learning_rate,
            'Epochs': self.args.epochs,
            'Early Stopping': self.args.early_stopping,
            'Reduce LR': self.args.reduce_lr,
            'Padding': self.args.padding,
            'Batch size': self.args.batch_size,
            'Use Original': self.args.use_original,
            'N mels': self.args.n_mels,
            'N FFT': self.args.n_fft,
            'Hop Length': self.args.hop_length,
            'Segment Length': self.args.segment_length,
            'Class Names': self.args.class_names,
            'Num Classes': self.args.num_classes}
        
        yaml_file = os.path.join(path_to_run, f'{os.path.basename(path_to_run)}.yaml')

        if not os.path.exists(yaml_file):   
            with open(yaml_file, 'w') as file:
                yaml.dump(current_config, file, default_flow_style=False)

    def load(self, args):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_file = os.path.join(script_dir, 'config', args.cfg)

        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"No YAML file found at {yaml_file}")

        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        for key, value in config.items():
            setattr(args, key.lower().replace(" ", "_"), value)

        return args

class CSVFilePath:
    @staticmethod
    def get(name):
        """Get CSV file path"""
        complete_name = f'{name}_dataset_split.csv'
        cwd = os.path.join(os.getcwd(), 'data', 'dataset')
        csv_file_path = os.path.join(cwd, complete_name)
        return csv_file_path

class SegLenConverter:
    @staticmethod
    def parse_and_convert(segment_length_string, sampling_rate):
        """Parses the segment length argument and converts it to samples."""
        parts = segment_length_string.split()
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "Expected format: '<value> samps' or '<value> ms'")

        val, unit = parts
        try:
            val = int(val)
        except ValueError:
            raise argparse.ArgumentTypeError("Value must be an integer.")

        if unit == "samps":
            return val
        elif unit == "ms":
            return int((val / 1000) * sampling_rate)
        else:
            raise argparse.ArgumentTypeError(
                "Unrecognized unit. Use 'samps' or 'ms'.")
        
class CSVDataset:
    @staticmethod
    def get_names_and_nbr(csv_file_path):
        data = pd.read_csv(csv_file_path)
        data = data['label_name'].unique()
        class_names = sorted(data.tolist())
        num_classes = len(class_names)
        # print(f'Class names: {class_names}')
        return class_names, num_classes
    
class ParquetDataset:
    @staticmethod
    def get_train_samples(parquet_file_path):
        df_lazy = pl.scan_parquet(parquet_file_path)
        
        train_samples_lazy = df_lazy.filter(pl.col("set") == "train")
        
        train_samples = train_samples_lazy.collect()
        
        label_indices = train_samples['label_index'].to_numpy()
        sample_indices = np.arange(len(train_samples)) 
        
        max_label_index = label_indices.max()
        
        classified_samples = [[] for _ in range(max_label_index + 1)]
        for sample_index, label_index in zip(sample_indices, label_indices):
            classified_samples[label_index].append(sample_index)
        
        classified_samples = [np.array(class_list) for class_list in classified_samples]
        return classified_samples

class DatasetMaker:
    def __init__(self, args):
        self.data_dir = 'data'
        self.destination = os.path.join(self.data_dir, 'dataset')
        self.name = args.name
        self.train_path = os.path.join(self.data_dir, 'preprocessed', args.name, args.train_dir)
        self.test_path = os.path.join(self.data_dir, 'preprocessed', args.name, args.test_dir)
        self.val_path = os.path.join(self.data_dir, 'preprocessed', args.name, args.val_dir) if args.val_dir is not None else None
        self.csv_path = os.path.join(self.destination, f'{self.name}_dataset_split.csv')
        self.parquet_path = os.path.join(self.destination, f'{self.name}_dataset_split.parquet')

    def make(self):
        all_files = {'train': self._scan_dir(self.train_path), 'test': self._scan_dir(self.test_path)}
        if self.val_path is not None:
            all_files['val'] = self._scan_dir(self.val_path)
        else:
            all_files['val'] = {}
            
        label_to_index = {label: idx for idx, label in enumerate(sorted(all_files['train'].keys()))}

        with open(self.csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['file_path', 'label_name', 'label_index', 'set'])

            for set_type in ['train', 'test', 'val']:
                self._process_files(writer, all_files[set_type], label_to_index, set_type)

        self._convert_to_parquet()

    # def _scan_dir(self, path):
    #     files_per_label = {}
    #     for root, _, files in tqdm(os.walk(path), desc=f'Scanning {os.path.basename(path)}'):
    #         valid_files = [os.path.join(root, f) for f in files if f.lower().endswith('.pt')]
    #         if valid_files:
    #             label = os.path.basename(root)
    #             files_per_label[label] = valid_files
    #     return files_per_label

    def _scan_dir(self, path):
        files_per_label = {}
        for root, dirs, files in tqdm(os.walk(path), desc=f'Scanning {os.path.basename(path)}'):
            # Ignore les répertoires vides ou autres fichiers non pertinents
            valid_files = [os.path.join(root, f) for f in files if f.lower().endswith('.pt')]
            if valid_files:
                label = os.path.basename(root)
                if label not in files_per_label:
                    files_per_label[label] = []
                files_per_label[label].extend(valid_files)  # Ajoute les fichiers à la liste de la classe correspondante
        return files_per_label

    def _process_files(self, writer, files_per_label, label_to_index, set_type):
        for label, files in files_per_label.items():
            label_index = label_to_index[label]
            for file in files:
                writer.writerow([file, label, label_index, set_type])

    def _convert_to_parquet(self):
        try:
            pl.read_csv(self.csv_path).write_parquet(self.parquet_path)
            print(f"Dataset saved as CSV and Parquet in {self.destination}")
        except Exception as e:
            print(f"Error writing Parquet file: {e}")


class DatasetSplitter:
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join('data', 'raw')
        self.train_dir = os.path.join(self.base_dir, args.train_dir) 
        self.test_dir = os.path.join(self.base_dir, args.test_dir) 
        self.val_dir = os.path.join(self.base_dir, args.val_dir) if args.val_dir is not None else None
        self.val_ratio = args.val_ratio
        self.val_split = args.val_split

    def split(self):
        if not self.val_dir:
            print("Validation directory not specified, splitting data for validatation set...")
            val_dir = os.path.join('data', 'raw', 'val')
            DirectoryManager.ensure_dir_exists(val_dir)
            
            if self.val_split == 'train':
                source_dir = self.train_dir
            elif self.val_split == 'test':
                source_dir = self.test_dir
            else:
                raise ValueError("val_split must be 'train' or 'test'")
            
            self.args.val_dir = 'val'
            self._split_data(source_dir)
        else:
            print("Validation directory specified, abort dataset splitting...")
    
    def _split_data(self, source_dir):
        all_files = []
        labels = []
        
        print(f"Checking directory: {source_dir}")
        print(f"Contents: {os.listdir(source_dir)}")
        
        def process_directory(dir_path, current_class=None):
            """Fonction récursive pour traiter les répertoires"""
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if not os.path.isdir(item_path):
                    continue
                
                audio_files = [f for f in os.listdir(item_path) if f.endswith(('.wav', '.mp3', '.flac', '.aiff', '.aif'))]
                if audio_files:
                    print(f"\nProcessing class: {item}")
                    print(f"Class path: {item_path}")
                    print(f"Found {len(audio_files)} audio files")
                    all_files.extend([(f, item) for f in audio_files])
                    labels.extend([item] * len(audio_files))
                else:
                    print(f"\nFound bank: {item}")
                    process_directory(item_path)
        
        process_directory(source_dir)
        
        if not all_files:
            raise ValueError(f"No audio files found in {source_dir}.")
        
        print(f"\nTotal files found: {len(all_files)}")
        _, val_files = train_test_split(all_files, test_size=self.val_ratio, random_state=42, stratify=labels)
        print(f"Files selected for validation: {len(val_files)}")
        
        for file, class_name in val_files:
            found = False
            for root, _, files in os.walk(source_dir):
                if file in files and os.path.basename(root) == class_name:
                    src_path = os.path.join(root, file)
                    val_class_dir = os.path.join(self.base_dir, 'val', class_name)
                    DirectoryManager.ensure_dir_exists(val_class_dir)
                    
                    dst_path = os.path.join(val_class_dir, file)
                    shutil.move(src_path, dst_path)
                    print(f"Moved {os.path.relpath(src_path, source_dir)} to validation set")
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find {file} in class {class_name}")

class DatasetValidator:
    def __init__(self, csv_file):
        self.csv = csv_file

    def validate(self):
        """Validates that the train, test, and val sets have the same unique labels."""
        data = pd.read_csv(self.csv)

        train_labels = set(data[data['set'] == 'train']['label_name'].unique())
        test_labels = set(data[data['set'] == 'test']['label_name'].unique())
        val_labels = set(data[data['set'] == 'val']['label_name'].unique())

        if not (train_labels == test_labels == val_labels):
            raise ValueError(
                "Mismatch in labels between train, test, and val sets.")

        print("Label validation passed: All sets have the same labels.")

class DirectoryManager:
    @staticmethod
    def ensure_dir_exists(directory):
        """Ensures that the directory exists. If not, creates it."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'{directory} has been created.')

