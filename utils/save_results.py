import pandas as pd
import os
import csv
from seaborn import heatmap
import matplotlib.pyplot as plt

from utils import CSVDataset

class Dict2MDTable:
    @staticmethod
    def apply(d, key='Name', val='Value'):
        """Convert args dictionnary to markdown table"""
        rows = [f'| {key} | {val} |']
        rows += ['|--|--|']
        rows += [f'| {k} | {v} |' for k, v in d.items()]
        return "  \n".join(rows)

class ToDisk:
    @staticmethod
    def save(args, stacked_metrics, cm, csv_file_path, current_run):
        """Save the results to disk as a CSV file."""
        labels, _ = CSVDataset.get_names_and_nbr(csv_file_path)

        log_dir = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_log_dir = os.path.join(log_dir, os.path.basename(current_run))
        if not os.path.exists(current_log_dir):
            os.makedirs(current_log_dir)

        csv_path = os.path.join(current_log_dir, f'results_{os.path.basename(current_run)}.csv')

        acc, f1, loss = stacked_metrics.tolist()
        accuracy = '%.4f' % acc
        f1 = '%.4f' % f1
        loss = '%.8f' % loss

        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if write_header:
                writer.writerow(['Run Name',
                                 'Current Run',
                                 'Model Name',
                                 'Sample Rate',
                                #  'Segment Overlap',
                                 'Fmin',
                                 'Fmax',
                                 'Mel Bands',
                                 'FFT',
                                 'Hop Length',
                                 'Segment Length',
                                 'Learning Rate',
                                 'Epochs',
                                #  'Offline Augmentations',
                                 'Online Augmentations',
                                 'Early Stopping',
                                 'Reduce LR on Plateau',
                                #  'Padding',
                                 'Accuracy',
                                 'Precision',
                                 'Recall',
                                 'F1 Score',
                                 'Loss'])

            writer.writerow([args.name,
                             args.current_run_name,
                             args.model,
                             args.sampling_rate,
                            #  args.segment_overlap,
                             args.f_min,
                             args.f_max,
                             args.n_mels,
                             args.n_fft,
                             args.hop_length,
                             args.segment_length,
                             args.learning_rate,
                             args.epochs,
                            #  args.offline_augment,
                             args.online_augment,
                             args.early_stopping,
                             args.reduce_lr,
                            #  args.padding,
                             accuracy,
                             f1,
                             loss])

        cm_path = os.path.join(current_log_dir, f'cm_{os.path.basename(current_run)}.csv')
        cm_np = cm.cpu().numpy()
        df_cm = pd.DataFrame(cm_np, index=labels, columns=labels)
        df_cm.to_csv(cm_path)

class ToTensorboard:
    @staticmethod
    def upload(stacked_metrics, cm, csv_file_path, log_queue):
        acc, f1, loss = stacked_metrics.tolist()
        dictResult = {
            'Accuracy': '%.4f' % acc,
            'Macro F1 Score': '%.4f' % f1,
            'Loss': '%.8f' % loss
        }

        log_queue.put(('Text/Results', Dict2MDTable.apply(dictResult), 0))

        labels, _ = CSVDataset.get_names_and_nbr(csv_file_path)

        cm_np = cm.cpu().numpy()
        df_cm = pd.DataFrame(cm_np, index=labels, columns=labels)
        df_cm_normalized = df_cm.div(df_cm.sum(axis=1), axis=0) * 100

        plt.figure(figsize=(12, 7))
        plt.xticks(rotation=45)
        heatmap_fig = heatmap(df_cm_normalized, annot=True, fmt=".2f", cmap="Blues").get_figure()

        log_queue.put(('Figure/ConfusionMatrix', heatmap_fig, 0))
        plt.close(heatmap_fig)

        print("Results uploaded to TensorBoard queue")

