from torch.utils.data import DataLoader, Dataset
from externals.pytorch_balanced_sampler.sampler import SamplerFactory
import torch
import os
import polars as pl

from utils.dataset_manager import ParquetDataset

class PrepareData:
    """Prepare datasets in processing the audio samples from train, val, test dirs."""
    def __init__(self, args, csv_file_path):
        self.args = args
        self.csv = csv_file_path
        self.parquet = os.path.splitext(csv_file_path)[0] + '.parquet'

    def batch_sampler_method(self, class_samples):
        total_samples = sum(len(samples) for samples in class_samples)
        max_batches = total_samples // self.args.batch_size

        batch_sampler = SamplerFactory().get(
            class_idxs=class_samples,
            batch_size=self.args.batch_size,
            n_batches=max_batches,
            alpha=1,
            kind='fixed'
        )

        return batch_sampler
    
    def prepare(self):
        class_samples_parquet = ParquetDataset.get_train_samples(self.parquet)

        batch_sampler = self.batch_sampler_method(class_samples_parquet)

        # if self.args.online_augment:
        #     transform = self.apply_transform(self.args.sampling_rate, self.args.f_min)
        # else:
        #     transform = None

        df_parquet = pl.read_parquet(self.parquet).to_pandas()

        train_loader = DataLoader(CustomTrainDataset(df_parquet), batch_sampler=batch_sampler, num_workers=self.args.num_workers, prefetch_factor=2, pin_memory=True)
        test_loader = DataLoader(CustomTestDataset(df_parquet), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, prefetch_factor=2, pin_memory=True)
        val_loader = DataLoader(CustomValDataset(df_parquet), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, prefetch_factor=2, pin_memory=True)

        print('Data successfully loaded into DataLoaders.')

        return train_loader, test_loader, val_loader

class CustomTrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.segments = self.df[self.df['set'] == 'train']
        self.transform = transform

    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.segments['file_path'].iloc[idx]
        waveform = torch.load(file_path).unsqueeze(0)  # Load .pt file instead of audio
        label = torch.tensor(self.segments['label_index'].iloc[idx], dtype=torch.long)

        return waveform, label
    
    def apply_batch_transform(self, data, targets):
        """Apply transform to the entire batch"""
        if self.transform:
            return self.transform(data)
        return data, targets
    
class CustomTestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.segments = self.df[self.df['set'] == 'test'].reset_index(drop=True)

    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.segments['file_path'].iloc[idx]
        waveform = torch.load(file_path).unsqueeze(0) 
        label = torch.tensor(self.segments['label_index'].iloc[idx], dtype=torch.long)

        return waveform, label
class CustomValDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.segments = self.df[self.df['set'] == 'val'].reset_index(drop=True)

    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.segments['file_path'].iloc[idx]
        waveform = torch.load(file_path).unsqueeze(0) 
        label = torch.tensor(self.segments['label_index'].iloc[idx], dtype=torch.long)

        return waveform, label
