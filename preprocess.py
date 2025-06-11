from utils import *
import os, argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the dataset and save to disk a csv file.')
    parser.add_argument('-n', '--name', type=str, help='Name of the project.', required=True)
    parser.add_argument('-sr', '--sampling_rate', type=int, help='Sampling rate for downsampling the audio files.', default=44100)
    parser.add_argument('--train_dir', type=str, help='Directory of training samples to preprocess.', default='train')
    parser.add_argument('--test_dir', type=str, help='Directory of test samples to preprocess.', default='test')
    parser.add_argument('--val_dir', type=str, help='Directory of val samples to preprocess.')
    parser.add_argument('--val_split', type=str, help='Specify on which dataset the validation split would be made.')
    parser.add_argument('--val_ratio', type=float, help='Amount of validation samples.', default=0.2)
    parser.add_argument('--segment_overlap', type=str, help='Overlap between audio segments. Increase the data samples by a factor 2.')
    parser.add_argument('--padding', type=str, help='Pad the arrays with zeros.', default='minimal')
    parser.add_argument('--use_original', type=int, help='Use original data for training.', default=1)
    parser.add_argument('-seglen','--segment_length', type=str, help='Defines segment length of audio data samples.', default='14700 samps')
    parser.add_argument('-atk', '--attack', type=int, help='Use attack augmentations.', default=0)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dataset_dir = 'data/dataset'
    raw_dir = 'data/raw'
    preprocessed_dir = 'data/preprocessed'

    dataset_splitter = DatasetSplitter(args)
    dataset_splitter.split()

    args.segment_length = SegLenConverter.parse_and_convert(args.segment_length, args.sampling_rate)

    # check if there is the same dataset as input ex: 'ismir_A_dataset_split.csv'
        # if not start preprocessing audio, make a csv and a parquet
        # if yes check if there is a parquet version
            # if not make a parquet version of
   
    for split in [args.train_dir, args.test_dir, args.val_dir]:
        if split is not None:  # Only process if the directory is specified
            dir_path = os.path.join(raw_dir, split)
            preprocessor = AudioPreprocessor(args, dir_path, preprocessed_dir)
            preprocessor.process_directory()

    if args.attack == 1:
        attack_dir = os.path.join(preprocessed_dir, args.name) # args.train_dir
        attack_augmenter = AttackAugmenter(attack_dir)
        attack_augmenter.process_files()

    dataset_maker = DatasetMaker(args)
    dataset_maker.make()

    dataset_validator = DatasetValidator(os.path.join(dataset_dir, f'{args.name}_dataset_split.csv'))
    dataset_validator.validate()
