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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='This script launches the training process.')
    parser.add_argument('-n','--name', type=str, help='Name of the project.', required=True)
    parser.add_argument('-d','--device', type=str, help='Specify the hardware on which computation should be performed.', default='cpu')
    parser.add_argument('-m','--model', type=str, help='Model version.', default='v1')
    parser.add_argument('-sr','--sampling_rate', type=int, help='Sampling rate for downsampling the audio files.', default=44100)
    parser.add_argument('--f_min', type=int, help='Minimum frequency for logmelspec analysis.', default=20)
    parser.add_argument('--f_max', type=int, help='Maximum frequency for logmelspec analysis.')
    parser.add_argument('--online_augment', type=int, help='Specify which online augmentations to use.')
    parser.add_argument('-lr','--learning_rate', type=float, help='Learning rate.', default=0.001)
    parser.add_argument('-e', '--epochs', type=int, help='Number of training epochs.', default=100)
    parser.add_argument('--early_stopping', type=int, help='Number of epochs without improvement before early stopping.', default=None)
    parser.add_argument('--reduce_lr', type=int, help='Reduce learning rate if validation plateaus.', default=None)
    parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size.')
    parser.add_argument('--n_mels', type=int, help='Number of Mel bands.', default=128)
    parser.add_argument('--n_fft', type=int, help='FFT Window.', default=2048)
    parser.add_argument('--hop_length', type=int, help='Hop length.', default=512)
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading.', default=4)
    parser.add_argument('-seglen','--segment_length', type=str, help='Defines segment length of audio data samples.', default='14700 samps')
    parser.add_argument('-cfg', '--config', type=str, help='Specify YAML file path.')

    args = parser.parse_args()
    return args

def logger_thread(log_queue, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    while not training_done or not log_queue.empty():
        log_data = log_queue.get()
        if log_data is None:
            continue  # Ignore None si l'entraînement n'est pas fini
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

    config_file_manager = ConfigFile(args)

    if args.config:
        args = config_file_manager.load(args)
    else:
        args.segment_length = SegLenConverter.parse_and_convert(args.segment_length, args.sampling_rate)

    device = DeviceName.get(args.device)
    csv_file_path = CSVFilePath.get(args.name)
    args.class_names, args.num_classes = CSVDataset.get_names_and_nbr(csv_file_path)

    dataPreparator = PrepareData(args, csv_file_path)
    train_loader, test_loader, val_loader = dataPreparator.prepare()

    date = datetime.datetime.now().strftime('%Y%m%d')
    time = datetime.datetime.now().strftime('%H%M%S')
    current_run = f'{RunDirectory.get(args.name)}_{date}_{time}'
    config_file_manager.save(current_run)

    current_run_name = f'{args.name}_{date}_{time}'
    setattr(args, "current_run_name", current_run_name)
    print(f'Run {current_run_name}')

    modelPreparator = PrepareModel(args)
    model = modelPreparator.prepare()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    if args.reduce_lr:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    transform = AudioOnlineTransforms(args)

    trainer = ModelTrainer(model, loss_fn, args.device, transform)

    log_queue = Queue()
    log_dir = f'runs/{args.name}_{date}_{time}'
    threading.Thread(target=logger_thread, args=(log_queue, log_dir), daemon=True).start()

    args_dict = vars(args)
    log_queue.put(('Text/Hyperparameters', Dict2MDTable.apply(args_dict), 1))

    max_val_loss = np.inf
    early_stopping_threshold = args.early_stopping
    counter = 0
    num_epoch = 0

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer)

        log_queue.put(('epoch/epoch', epoch, epoch))
        log_queue.put(('Loss/train', train_loss, epoch))
        empty_cache()
        val_loss, val_acc, val_f1 = trainer.validate_epoch(val_loader)
        log_queue.put(('Loss/val', val_loss, epoch))
        log_queue.put(('Accuracy/val', val_acc, epoch))
        log_queue.put(('F1/val', val_f1, epoch))

        if args.reduce_lr:
            scheduler.step(val_loss)
        
        if val_loss < max_val_loss:
            max_val_loss = val_loss
            counter = 0
            num_epoch = epoch
            best_state = model.state_dict()
            # Save the checkpoints each time a model state is better than the last one
            torch.save(model.state_dict(), f'{current_run}/{args.name}_{date}_{time}.pth')
            empty_cache()
        else:
            if args.early_stopping:
                counter += 1
                if counter >= early_stopping_threshold:
                    print('Early stopping triggered.')
                    break
    
    model.load_state_dict(best_state)

    stkd_mtrs, cm = trainer.test_model(test_loader)

    ToTensorboard.upload(stkd_mtrs, cm, csv_file_path, log_queue)
    print(f'Results and Confusion Matrix have been uploaded to tensorboard.')

    ToDisk.save(args, stkd_mtrs, cm, csv_file_path, current_run)
    print(f'Results and Confusion Matrix have been saved in the logs/{os.path.basename(current_run)} directory.')

    torch.save(model.state_dict(), f'{current_run}/{args.name}_{date}_{time}.pth')
    print(f'Checkpoints has been saved in the {os.path.relpath(current_run)} directory.')

    model = model.to('cpu')
    scripted_model = script(model)
    scripted_model.save(f'{current_run}/{args.name}_{date}_{time}.ts')
    print(f'TorchScript file has been exported to the {os.path.relpath(current_run)} directory.')

    log_queue.put(None)
    os._exit(0)