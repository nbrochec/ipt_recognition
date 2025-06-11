import torch
import torch.nn as nn

from models.layers import LogMelSpectrogramLayer, customConv2d

class eguitar(nn.Module):
    def __init__(self, output_nbr, args):
        super(eguitar, self).__init__()

        self.sr = args.sampling_rate
        self.classnames = args.class_names
        self.fmin = args.f_min
        self.fmax = args.f_max
        self.segment_length = args.segment_length

        self.nmels = args.n_mels
        self.hoplength = args.hop_length

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=self.nmels, n_fft=2048, hop_length=self.hoplength)

        self.cnn = self._create_cnn_block()
        self.fc = self._create_fc_block(output_nbr)

    def _create_fc_block(self, output_nbr):
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout1d(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout1d(0.25),
            nn.Linear(128, output_nbr),
        )

    def _create_cnn_block(self):
         return nn.Sequential(
            nn.AdaptiveAvgPool2d((128, 15)),
            customConv2d(1, 64, (2,3), "same"),
            customConv2d(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customConv2d(64, 128, (2,3), "same"),
            customConv2d(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            customConv2d(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customConv2d(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            customConv2d(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customConv2d(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            customConv2d(512, 512, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

    @torch.jit.export
    def get_sr(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames
    
    @torch.jit.export
    def get_seglen(self):
        return self.segment_length

    def forward(self, x):
        x_logmel = self.logmel(x)
        x_cnn = self.cnn(x_logmel)
        x_flat = x_cnn.view(x_cnn.size(0), -1)
        z = self.fc(x_flat)
        return z

class flute(nn.Module):
    def __init__(self, output_nbr, args):
        super(flute, self).__init__()

        self.sr = args.sampling_rate
        self.classnames = args.class_names
        self.fmin = args.f_min
        self.fmax = args.f_max
        self.segment_length = args.segment_length

        self.nmels = args.n_mels
        self.hoplength = args.hop_length

        self.logmel1 = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=self.nmels, n_fft=512, hop_length=self.hoplength)
        self.logmel2 = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=self.nmels, n_fft=1024, hop_length=self.hoplength)
        self.logmel3 = LogMelSpectrogramLayer(sample_rate=self.sr, f_min=self.fmin, f_max=self.fmax, n_mels=self.nmels, n_fft=2048, hop_length=self.hoplength)
        
        self.cnn = self._create_cnn_block()
        self.fc = self._create_fc_block(output_nbr)

    def _create_fc_block(self, output_nbr):
        return nn.Sequential(
                nn.Linear(160, 80),
                nn.BatchNorm1d(80),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(80, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout1d(0.25),
                nn.Linear(40, output_nbr)
            )

    def _create_cnn_block(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((384, 112)),
            customConv2d(3, 40, 4, "same"), 
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(0.25),
            customConv2d(40, 80, 3, "same"),
            nn.MaxPool2d((4, 2)), 
            nn.Dropout2d(0.25),
            customConv2d(80, 160, 2, "same"),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),
            customConv2d(160, 160, 2, "same"),
            nn.MaxPool2d((12, 14)),
            nn.Dropout2d(0.25),
        )
    
    @torch.jit.export
    def get_sr(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames
    
    @torch.jit.export
    def get_seglen(self):
        return self.segment_length

    def forward(self, x):
        x1 = self.logmel1(x)
        x2 = self.logmel2(x)
        x3 = self.logmel3(x)

        x_concat = torch.cat((x1, x2, x3), dim=1)

        x_cnn = self.cnn(x_concat) 
        x_flat = x_cnn.view(x_cnn.size(0), -1)
        z = self.fc(x_flat)
        return z 
