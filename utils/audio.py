from librosa.effects import pitch_shift, time_stretch, trim
from librosa import A4_to_tuning, load, resample
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import hashlib
import torch
from tqdm import tqdm
from glob import glob
from torch_audiomentations import Compose, PolarityInversion, HighPassFilter, LowPassFilter

class AudioPreprocessor:
    def __init__(self, args, input_dir, output_dir):
        self.name = args.name
        self.input_dir = input_dir
        self.output_dir = os.path.join(output_dir, args.name, os.path.basename(input_dir))
        self.target_sr = args.sampling_rate
        self.segment_length = args.segment_length
        self.segment_overlap = args.segment_overlap
        self.use_original = args.use_original
        self.padding = args.padding
        self.augmenter = AudioOfflineTransforms(args.sampling_rate)
        
    def ensure_dir_exists(self, directory):
        """Ensure the folder exists"""
        os.makedirs(directory, exist_ok=True)

    def pad_waveform(self, waveform, target_length):
        """Add silence to the waveform to match the target length."""
        extra_length = target_length - len(waveform)
        if extra_length > 0:
            silence = np.zeros(extra_length)
            waveform = np.concatenate((waveform, silence))
        return waveform

    def process_audio_file(self, file_path, output_dir):
        """Load audio file, convert to target sr, segment it, apply augmentations, and save."""
        try:
            waveform, original_sr = load(file_path, sr=None)

            if original_sr != self.target_sr:
                waveform = resample(waveform, orig_sr=original_sr, target_sr=self.target_sr)

            waveform, _ = trim(waveform)

            if self.padding == 'minimal' and len(waveform) < self.segment_length:
                waveform = self.pad_waveform(waveform, self.segment_length)

            segments = self.process_segments(waveform)

            for idx, segment in enumerate(segments):
                if 'train' in file_path:
                    aug1, aug2, aug3 = self.augmenter(segment)
                    self.save_segment(aug1, output_dir, idx, "detuned", file_path)
                    self.save_segment(aug2, output_dir, idx, "noise", file_path)
                    self.save_segment(aug3, output_dir, idx, "stretched", file_path)
                    # self.save_segment(aug4, output_dir, idx, "shifted", file_path)
                
                self.save_segment(segment, output_dir, idx, "original", file_path)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def process_segments(self, waveform):
        """Process the waveform by dividing it into segments with or without overlap."""
        segments = []
        num_samples = len(waveform)
        step = self.segment_length // 2 if self.segment_overlap else self.segment_length
        
        for i in range(0, num_samples, step):
            if i + self.segment_length <= num_samples:
                segment = waveform[i:i + self.segment_length]
            else:
                if self.padding == 'full':
                    valid_length = num_samples - i
                    segment = np.zeros(self.segment_length)
                    segment[:valid_length] = waveform[i:i + valid_length]
                else:
                    continue  # Skip segments that are too short
            segments.append(segment)
        
        return segments

    def save_segment(self, segment, output_dir, index, aug_type, file_path):
        """Save the processed segment as a PyTorch tensor."""
        base_name = os.path.basename(os.path.splitext(file_path)[0])
        segment_hash = hashlib.sha256(segment).hexdigest()[:8]
        self.ensure_dir_exists(output_dir)
        
        filename = f"{base_name}_segment{index}_{aug_type}_{segment_hash}.pt"
        output_path = os.path.join(output_dir, filename)
        
        segment_tensor = torch.tensor(segment, dtype=torch.float32)
        torch.save(segment_tensor, output_path)

    def process_directory(self):
        """Retrieve input folder and process the audio files."""
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        total_files = sum(
            len(files) for _, _, files in os.walk(self.input_dir)
            if any(file.lower().endswith(('.wav', '.mp3', '.flac', '.aiff', '.aif')) for file in files)
        )

        print(f"Total audio files to process: {total_files}")

        audio_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.aiff', '.aif')):
                    file = file.replace(" ", "_")
                    input_file = os.path.join(root, file)
                    audio_files.append(input_file)

        with tqdm(total=total_files, desc="Preprocessing audio files") as pbar:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.process_audio_file, file, os.path.join(self.output_dir, os.path.relpath(os.path.dirname(file), self.input_dir))): file for file in audio_files}
                for future in futures:
                    future.add_done_callback(lambda p: pbar.update(1))

        print("Preprocessing done.")

class AttackAugmenter:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def find_target_files(self):
        """Look for '.pt' files with 'segment0' in their file name."""
        pattern = os.path.join(self.root_dir, "**", "*segment0*.pt")
        print(pattern)
        return glob(pattern, recursive=True)

    def augment_tensor(self, tensor):
        """Add 0 to tensor"""
        original_length = tensor.shape[0]
        augmented_tensors = []
        
        for ratio in [0.25, 0.5, 0.75]:
            pad_length = int(original_length * ratio)
            padding = torch.zeros(pad_length)
            augmented_tensor = torch.cat((padding, tensor))[:original_length]
            augmented_tensors.append(augmented_tensor)
        
        return augmented_tensors

    def process_files(self):
        """Process and save new files."""
        files = self.find_target_files()
        
        with tqdm(total=len(files), desc="Applying attack augmentation") as pbar:
            for file_path in files:
                tensor = torch.load(file_path)
                augmented_tensors = self.augment_tensor(tensor)
                
                for i, aug_tensor in enumerate(augmented_tensors):
                    aug_file_path = file_path.replace(".pt", f"_shifted{i}.pt")
                    torch.save(aug_tensor, aug_file_path)
                    pbar.update(1)
                    # print(f"Saved: {aug_file_path}")

class AudioOfflineTransforms:
    def __init__(self, target_sr):
        self.sr = target_sr

    def custom_detune(self, data):
        """Apply detuning."""
        random_tuning = np.random.uniform(-100, 100) + 440
        change_tuning = A4_to_tuning(random_tuning)
        return pitch_shift(data, sr=self.sr, n_steps=change_tuning, bins_per_octave=12)

    def custom_gaussnoise(self, data):
        """Add Gaussian noise."""
        noise = np.random.normal(0, 0.01, size=data.shape)
        return data + noise
    
    def lb_timestretch(self, data):
        """Apply stretch."""
        rate = np.random.uniform(0.9, 1.1)
        return time_stretch(data, rate=rate)
    
    def custom_shift(self, data):
        nbr = len(data)//2
        return np.roll(data, nbr)

    def pad_or_trim(self, data, original_size):
        """Pad or trim"""
        current_size = data.shape[0]
        if current_size > original_size:
            return data[:original_size]
        elif current_size < original_size:
            return np.pad(data, pad_width=(0, original_size - current_size), mode='constant', constant_values=0)
        return data

    def __call__(self, data):
        """Apply audio augmentation."""
        original_size = len(data)
        if data.ndim > 1:
            data = data.squeeze(0)

        aug_detuned = self.custom_detune(data)
        aug_noised = self.custom_gaussnoise(data)
        aug_stretched = self.lb_timestretch(data)
        # aug_shift = self.custom_shift(data)

        aug_detuned = self.pad_or_trim(aug_detuned, original_size)
        aug_noised = self.pad_or_trim(aug_noised, original_size)
        aug_stretched = self.pad_or_trim(aug_stretched, original_size)
        # aug_shift = self.custom_shift(aug_shift, original_size)

        return aug_detuned, aug_noised, aug_stretched #aug_shift

class AudioOnlineTransforms():
    def __init__(self, args):
        self.fmin = args.f_min
        self.sr = args.sampling_rate
        self.online_augment = args.online_augment if args.online_augment else None

        self.compose = Compose(
                transforms=[
                    PolarityInversion(p=0.5, output_type='tensor'),
                    HighPassFilter(p=0.5, output_type='tensor', min_cutoff_freq=self.fmin, max_cutoff_freq=self.fmin*2, sample_rate=self.sr),
                    LowPassFilter(p=0.5, output_type='tensor', min_cutoff_freq=self.sr//4, max_cutoff_freq=(self.sr//2)-100, sample_rate=self.sr)
                    # Shift(p=0.5, output_type='tensor', rollover=True),
                    # Shift(p=0.5, output_type='tensor', rollover=False),
                ]
            )

    def __call__(self, data, targets):
        if self.online_augment:
            data = self.compose(data)
        return data, targets