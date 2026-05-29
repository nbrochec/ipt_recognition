import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import os
import torch
import numpy as np
import datetime
import threading
from queue import Queue
from torch.cuda import empty_cache
from torch.jit import script

from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from models import *
from utils import *

def parse_arguments():
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description='Fine-tune a pretrained model.')
    
    # Required arguments
    parser.add_argument('-n','--name', type=str, help='Name of the project.', required=True)
    parser.add_argument('-pt', '--pretrained', type=str, help='Path to pretrained .pth model file.', default='pretrained/flute_pretrained.pth')
    
    # Model configuration
    parser.add_argument('-d','--device', type=str, help='Specify the hardware on which computation should be performed.', default='cpu')
    parser.add_argument('-m','--model', type=str, help='Model architecture (must match pretrained model).', default='base')
    
    # Audio processing
    parser.add_argument('-sr','--sampling_rate', type=int, help='Sampling rate for downsampling the audio files.', default=44100)
    parser.add_argument('--f_min', type=int, help='Minimum frequency for logmelspec analysis.', default=20)
    parser.add_argument('--f_max', type=int, help='Maximum frequency for logmelspec analysis.')
    parser.add_argument('--n_mels', type=int, help='Number of Mel bands.', default=384)
    parser.add_argument('--n_fft', type=int, help='FFT Window.', default=2048)
    parser.add_argument('--hop_length', type=int, help='Hop length.', default=128)
    parser.add_argument('-seglen','--segment_length', type=str, help='Defines segment length of audio data samples.', default='14700 samps')
    
    # Fine-tuning specific
    parser.add_argument('--freeze_conv', type=int, help='Freeze convolutional layers (1) or train all layers (0).', default=1)
    parser.add_argument('--online_augment', type=int, help='Use online augmentations based on polarity inversion, lpf and hpf.')
    
    # Training hyperparameters
    parser.add_argument('-lr','--learning_rate', type=float, help='Learning rate (use lower for fine-tuning).', default=0.0003)
    parser.add_argument('-e', '--epochs', type=int, help='Number of training epochs.', default=50)
    parser.add_argument('--early_stopping', type=int, help='Number of epochs without improvement before early stopping.', default=5)
    parser.add_argument('--reduce_lr', type=int, help='Reduce learning rate if validation plateaus.', default=1)
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size.')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading.', default=4)
    
    # Config file
    parser.add_argument('-cfg', '--config', type=str, help='Specify YAML file path.')

    args = parser.parse_args()
    return args

def freeze_conv_layers(model, model_type):
    """
    Freeze convolutional layers based on model architecture.
    Only train the fully connected (classification) layers.
    """
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        # Freeze everything except fc layers
        if 'fc' not in name and 'logmel' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING CONFIGURATION:")
    print(f"{'='*60}")
    print(f"Frozen parameters:     {frozen_params:,} (conv layers + logmel)")
    print(f"Trainable parameters:  {trainable_params:,} (fc layers only)")
    print(f"{'='*60}\n")
    
    return model

def verify_audio_parameters(model, args):
    """
    Verify that audio parameters match between pretrained model and current args.
    Raises an error if critical parameters don't match.
    """
    print(f"\n{'='*60}")
    print("VERIFYING AUDIO PARAMETERS")
    print(f"{'='*60}")
    
    mismatches = []
    warnings = []
    
    # Check sampling rate
    if hasattr(model, 'sr') and model.sr != args.sampling_rate:
        mismatches.append(f"Sampling Rate: model={model.sr}, args={args.sampling_rate}")
    
    # Check segment length
    if hasattr(model, 'segment_length') and model.segment_length != args.segment_length:
        mismatches.append(f"Segment Length: model={model.segment_length}, args={args.segment_length}")
    
    # Check mel parameters
    if hasattr(model, 'nmels') and model.nmels != args.n_mels:
        warnings.append(f"N Mels: model={model.nmels}, args={args.n_mels}")
    
    if hasattr(model, 'hoplength') and model.hoplength != args.hop_length:
        warnings.append(f"Hop Length: model={model.hoplength}, args={args.hop_length}")
    
    if hasattr(model, 'fmin') and model.fmin != args.f_min:
        warnings.append(f"F Min: model={model.fmin}, args={args.f_min}")
    
    if hasattr(model, 'fmax') and args.f_max and model.fmax != args.f_max:
        warnings.append(f"F Max: model={model.fmax}, args={args.f_max}")
    
    # Print results
    if mismatches:
        print("\n CRITICAL MISMATCHES FOUND:")
        for mismatch in mismatches:
            print(f"  • {mismatch}")
        print("\n  Fine-tuning will likely fail. Please ensure:")
        print("   1. Preprocess your new data with the SAME parameters as the pretrained model")
        print("   2. Use the same -sr and -seglen arguments")
        print(f"\n{'='*60}\n")
        raise ValueError("Audio parameter mismatch detected. Cannot proceed with fine-tuning.")
    
    if warnings:
        print("\n  WARNINGS (non-critical):")
        for warning in warnings:
            print(f"  • {warning}")
        print("\nThese differences may affect model performance.")
        print("Consider using the same mel parameters as the pretrained model.")
    
    if not mismatches and not warnings:
        print("✓ All audio parameters match!")
    
    print(f"{'='*60}\n")

def load_pretrained_model(model, pretrained_path, num_classes, args):
    """
    Load pretrained weights for all matching layers (encoder + FC).
    Layers with mismatched shapes are skipped and keep their random initialization.
    Then we'll train on top of these weights.
    """
    print(f"\nLoading pretrained model from: {pretrained_path}")
    
    # Load pretrained checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Get model's current state dict
    model_dict = model.state_dict()
    
    # Load ALL matching layers (encoder + FC if shapes match)
    pretrained_dict = {}
    skipped_layers = []
    
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                skipped_layers.append(f"{k}: shape mismatch - pretrained={v.shape}, current={model_dict[k].shape}")
        else:
            skipped_layers.append(f"{k}: not in current model")
    
    # Update model with all pretrained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    loaded_encoder = len([k for k in pretrained_dict.keys() if 'fc' not in k])
    loaded_fc = len([k for k in pretrained_dict.keys() if 'fc' in k])
    
    print(f"✓ Loaded {loaded_encoder} encoder layers")
    print(f"✓ Loaded {loaded_fc} FC layers")
    
    if skipped_layers:
        print(f"\n⚠ Skipped {len(skipped_layers)} layer(s) (will use random init):")
        for layer in skipped_layers[:5]:
            print(f"  • {layer}")
        if len(skipped_layers) > 5:
            print(f"  ... and {len(skipped_layers) - 5} more")
    
    print(f"\n✓ Ready to fine-tune on {num_classes} classes\n")
    
    return model

def logger_thread(log_queue, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    while not training_done or not log_queue.empty():
        log_data = log_queue.get()
        if log_data is None:
            continue
        tag, value, step = log_data
        if tag.startswith('Text/'):
            writer.add_text(tag, value, step)
        elif tag.startswith('Figure/'):
            writer.add_figure(tag, value, step)
        else:
            writer.add_scalar(tag, value, step)
    writer.close()

training_done = False

if __name__ == '__main__':

    args = parse_arguments()
    checker = ModelChecker(args.model)

    # Load config if provided
    config_file_manager = ConfigFile(args)
    if args.config:
        args = config_file_manager.load()
    else:
        args.segment_length = SegLenConverter.parse_and_convert(args.segment_length, args.sampling_rate)

    # Check if pretrained model exists
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained model not found at: {args.pretrained}")

    device = DeviceName.get(args.device)
    csv_file_path = CSVFilePath.get(args.name)
    args.class_names, args.num_classes = CSVDataset.get_names_and_nbr(csv_file_path)

    # Prepare data
    dataPreparator = PrepareData(args, csv_file_path)
    train_loader, val_loader = dataPreparator.prepare()

    # Create run directory
    date = datetime.datetime.now().strftime('%Y%m%d')
    time = datetime.datetime.now().strftime('%H%M%S')
    current_run = f'{RunDirectory.get(args.name)}_finetune_{date}_{time}'
    config_file_manager.save(current_run)

    current_run_name = f'{args.name}_finetune_{date}_{time}'
    setattr(args, "current_run_name", current_run_name)
    print(f'Fine-tuning Run: {current_run_name}\n')

    # Initialize model
    modelPreparator = PrepareModel(args)
    model = modelPreparator.prepare()

    # Load pretrained weights
    model = load_pretrained_model(model, args.pretrained, args.num_classes, args)

    # Verify audio parameters match
    verify_audio_parameters(model, args)

    # Freeze convolutional layers if requested
    if args.freeze_conv:
        model = freeze_conv_layers(model, args.model)

    model = model.to(device)

    # Setup optimizer - only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.learning_rate, weight_decay=1e-5)

    print(f"\nOptimizing {len(trainable_params)} parameter groups")
    print(f"Learning rate: {args.learning_rate}\n")

    loss_fn = nn.CrossEntropyLoss()

    if args.reduce_lr:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    transform = AudioOnlineTransforms(args)
    trainer = ModelTrainer(model, loss_fn, args.device, transform, args.num_classes)

    # Setup logging
    log_queue = Queue()
    log_dir = f'runs/{args.name}_finetune_{date}_{time}'
    threading.Thread(target=logger_thread, args=(log_queue, log_dir), daemon=True).start()

    args_dict = vars(args)
    log_queue.put(('Text/Hyperparameters', Dict2MDTable.apply(args_dict), 1))

    # Training loop
    max_val_loss = np.inf
    early_stopping_threshold = args.early_stopping
    counter = 0
    num_epoch = 0

    print(f"{'='*60}")
    print(f"STARTING FINE-TUNING")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer)

        log_queue.put(('epoch/epoch', epoch, epoch))
        log_queue.put(('Loss/train', train_loss, epoch))
        empty_cache()
        
        val_loss, val_acc, val_f1 = trainer.validate_epoch(val_loader)
        log_queue.put(('Loss/val', val_loss, epoch))
        log_queue.put(('Accuracy/val', val_acc, epoch))
        log_queue.put(('F1/val', val_f1, epoch))

        # print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

        if args.reduce_lr:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            log_queue.put(('Learning Rate', current_lr, epoch))
        
        if val_loss < max_val_loss:
            max_val_loss = val_loss
            counter = 0
            num_epoch = epoch
            best_state = model.state_dict()
            torch.save(model.state_dict(), f'{current_run}/{args.name}_finetune_{date}_{time}.pth')
            # print(f"  New best model saved (val_loss: {val_loss:.4f})")
            empty_cache()
        else:
            if args.early_stopping:
                counter += 1
                print(f"  Early stopping counter: {counter}/{early_stopping_threshold}")
                if counter >= early_stopping_threshold:
                    print('Early stopping triggered.')
                    break
    
    # Load best model and test
    model.load_state_dict(best_state)
    print(f"\n{'='*60}")
    print(f"TESTING BEST MODEL (from epoch {num_epoch+1})")
    print(f"{'='*60}\n")

    # stkd_mtrs, cm = trainer.test_model(test_loader)

    # ToTensorboard.upload(stkd_mtrs, cm, csv_file_path, log_queue)
    # print(f'Results and Confusion Matrix have been uploaded to tensorboard.')

    # ToDisk.save(args, stkd_mtrs, cm, csv_file_path, current_run)
    # print(f'Results and Confusion Matrix have been saved in the logs/{os.path.basename(current_run)} directory.')

    # Save final model
    torch.save(model.state_dict(), f'{current_run}/{args.name}_finetune_{date}_{time}.pth')
    print(f'Final checkpoint has been saved in the {os.path.relpath(current_run)} directory.')

    # Export TorchScript
    model = model.to('cpu')
    scripted_model = script(model)
    scripted_model.save(f'{current_run}/{args.name}_finetune_{date}_{time}.ts')
    print(f'TorchScript file has been exported to the {os.path.relpath(current_run)} directory.')

    print(f"\n{'='*60}")
    print(f"✓ FINE-TUNING COMPLETE!")
    print(f"{'='*60}\n")

    log_queue.put(None)
    training_done = True
    os._exit(0)