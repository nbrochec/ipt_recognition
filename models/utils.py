import torch
import humanize
import sys
import os

from models import *
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
     
class DeviceName:
    @staticmethod
    def get(selected_device):
        """Select the device."""
        if selected_device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print('This script uses CUDA as the torch device.')
        elif selected_device == 'cpu':
            device = torch.device('cpu')
            print('This script uses CPU as the torch device.')
        else:
            raise ValueError(f"Invalid or unavailable device: {selected_device}")

        return device

class ModelChecker:
    """Check if the provided model name is valid."""
    def __init__(self, model_name):
        self.available_models = {
            'eguitar', 'flute'
        }
        self.model_name = model_name
        self.validate_model()

    def validate_model(self):
        """Validate the model name."""
        if self.model_name not in self.available_models:
            available = ', '.join(self.available_models)
            raise ValueError(
                f"Error: Model '{self.model_name}' is not recognized. Available models: {available}."
            )
        print(f"Model '{self.model_name}' is valid. Continuing execution...")

class LoadModel:
    """A class for loading different model architectures based on a provided model name."""

    def __init__(self):
        self.models = {
            'eguitar': eguitar,
            'flute': flute,
        }

    def get_model(self, model_name, output_nbr, args):
        if model_name in self.models:
            return self.models[model_name](output_nbr, args)
        else:
            raise ValueError(f"Model {model_name} is not recognized.")


class ModelSummary:
    def __init__(self, model, num_labels, config):
        """Initializes the ModelSummary class."""
        self.model = model
        self.num_labels = num_labels
        self.config = config

    def get_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def print_summary(self):
        total_params = self.get_total_parameters()
        formatted_params = humanize.intcomma(total_params)

        print('\n')
        print('-----------------------------------------------')
        print(f"Model Summary:")
        print(f"Model's name: {self.config}")
        print(f"Number of labels: {self.num_labels}")
        print(f"Total number of parameters: {formatted_params}")
        print('-----------------------------------------------')
        print('\n')


class ModelTester:
    def __init__(self, model, input_shape=(1, 1, 7680), device='cpu'):
        """Initializes the ModelTester class."""
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def test(self):
        """Tests the model with a random input tensor."""
        self.model.to(self.device)
        self.model.eval()

        random_input = torch.randn(self.input_shape).to(self.device)

        with torch.no_grad():
            output = self.model(random_input)

        return output


class ModelInit:
    def __init__(self, model):
        """Initialize the ModelInit class. """
        self.model = model

    def initialize(self):
        """Apply weight initialization to the model layers."""
        init_method = torch.nn.init.xavier_normal_

        for layer in self.model.modules():
            if isinstance(
                    layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
                init_method(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        return self.model


class ModelTrainer:
    def __init__(self, model, loss_fn, device, transform):
        """Initialize the ModelTrainer."""
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.transform = transform

    def train_epoch(self, loader, optimizer):
        """Perform one training epoch."""
        self.model.train()
        running_loss = 0.0

        for data, targets in tqdm(loader, desc="Training", leave=False):
            data, targets = self.transform(data.to(self.device), targets.to(self.device))
            
            optimizer.zero_grad()

            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        return running_loss / len(loader.dataset)
    
    def validate_epoch(self, loader):
        """Perform one validation epoch."""
        self.model.eval()
        running_loss = 0.0

        all_targets = []
        for data, targets in loader:
            all_targets.append(targets.cpu())
        all_targets = torch.cat(all_targets)
        class_nbr = len(torch.unique(all_targets))

        val_acc = MulticlassAccuracy(
            num_classes=class_nbr,
            average='micro').to(
            self.device)
        val_f1 = MulticlassF1Score(
            num_classes=class_nbr,
            average='macro').to(
            self.device)

        with torch.no_grad():
            for data, targets in tqdm(loader, desc="Validation", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)

                val_acc.update(preds=predicted, target=targets)
                val_f1.update(preds=predicted, target=targets)

        val_loss = running_loss / len(loader.dataset)
        val_loss = torch.tensor(val_loss).to(self.device)

        val_acc = val_acc.compute().detach().cpu().item()
        val_f1 = val_f1.compute().detach().cpu().item()

        return val_loss, val_acc, val_f1

    def test_model(self, loader):
        """Test the model and compute metrics."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_targets = []
        for data, targets in loader:
            all_targets.append(targets.cpu())
        all_targets = torch.cat(all_targets)
        class_nbr = len(torch.unique(all_targets))

        accuracy_metric = MulticlassAccuracy(
            num_classes=class_nbr,
            average='micro').to(
            self.device)
        f1_metric = MulticlassF1Score(
            num_classes=class_nbr,
            average='macro').to(
            self.device)
        cm_metric = MulticlassConfusionMatrix(
            num_classes=class_nbr).to(
            self.device)

        with torch.no_grad():
            for data, targets in tqdm(loader, desc="Test", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets)
                batch_size = data.size(0)
                _, predicted = torch.max(outputs, 1)

                accuracy_metric.update(preds=predicted, target=targets)
                f1_metric.update(preds=predicted, target=targets)
                cm_metric.update(preds=predicted, target=targets)

                running_loss += loss.item() * batch_size
                total_samples += batch_size

        accuracy = torch.round(accuracy_metric.compute(), decimals=6) * 100
        f1 = torch.round(f1_metric.compute(), decimals=6) * 100
        cm = cm_metric.compute()

        running_loss = running_loss / total_samples

        print(f'Test Loss: {running_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test Macro F1 Score: {f1:.4f}')

        stacked_metrics = torch.stack([accuracy.to(self.device), f1.to(self.device), torch.tensor(running_loss).to(self.device)], dim=0)

        return stacked_metrics, cm


class ModelSaver:
    """Class to save PyTorch model checkpoints and export TorchScript models."""
    @staticmethod
    def save(model, current_run, args, date, time):
        os.makedirs(current_run, exist_ok=True)

        checkpoint_path = os.path.join(
            current_run, f'{args.name}_{date}_{time}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Checkpoint has been saved in the {os.path.relpath(current_run)} directory.")

        # Export TorchScript
        model = model.to('cpu')
        scripted_model = torch.jit.script(model)
        torchscript_path = os.path.join(
            current_run, f"{args.name}_{date}_{time}.ts")
        scripted_model.save(torchscript_path)
        print(
            f"TorchScript file has been exported to the {os.path.relpath(current_run)} directory.")

        # Return 
        return checkpoint_path, torchscript_path


class PrepareModel:
    def __init__(self, args):
        """Initialize the PrepareModel class."""
        self.args = args
        self.num_classes = args.num_classes
        self.seglen = args.segment_length
        self.device = args.device

    def prepare(self):
        model = LoadModel().get_model(
            self.args.model,
            self.num_classes,
            self.args).to(
            self.device)

        tester = ModelTester(
            model,
            input_shape=(
                1,
                1,
                self.seglen),
            device=self.device)
        output = tester.test()
        if output.size(1) != self.num_classes:
            print("Error: Output dimension does not match the number of classes.")
            sys.exit(1)

        summary = ModelSummary(model, self.num_classes, self.args.model)
        summary.print_summary()

        model = ModelInit(model).initialize()
        return model


class TestSavedModel:
    def __init__(self, model_name, output_nbr, args, model_dir):
        """Initialize the TestSavedModel class."""
        self.model_name = model_name
        self.output_nbr = output_nbr
        self.args = args
        self.model_dir = model_dir
        self.model_loader = LoadModel()

    def get_latest_checkpoint(self):
        """Find the last model's checkpoint"""
        pth_files = [
            f for f in os.listdir(
                self.model_dir) if f.endswith('.pth')]
        if not pth_files:
            raise ValueError("No model checkpoint found.")
        latest_file = max(
            pth_files,
            key=lambda x: os.path.getmtime(
                os.path.join(
                    self.model_dir,
                    x)))
        return os.path.join(self.model_dir, latest_file)

    def load_model(self):
        """Load last saved model."""
        latest_checkpoint = self.get_latest_checkpoint()
        model = self.model_loader.get_model(
            self.model_name, self.output_nbr, self.args)
        model.load_state_dict(torch.load(latest_checkpoint))
        model.eval()
        return model

    def test_model(self, test_loader, device):
        """Test model with test_lodaer"""
        loss_fn = torch.nn.CrossEntropyLoss()
        model = self.load_model().to(device)
        trainer = ModelTrainer(model, loss_fn, device)
        stkd_mtrs, cm = trainer.test_model(test_loader)
        return stkd_mtrs, cm

    def export_model_to_ts(self, run_name):
        """Export model to ts"""
        model_ts_path = os.path.join(self.model_dir, f'{run_name}.ts')
        self.model.to('cpu')
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(model_ts_path)
        print(
            f'TorchScript model has been saved at {os.path.relpath(model_ts_path)}.')
