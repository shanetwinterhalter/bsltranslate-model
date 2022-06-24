import torch.nn as nn

from torch import load, double, flatten
from torch.hub import load as hub_load


class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                   stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.ReLU()(x)


class Cnn_3d(nn.Module):
    def __init__(self, n_y_vals, in_channels=1):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock3d(64, 64, downsample=False),
            ResBlock3d(64, 64, downsample=False),
        )

        self.layer2 = nn.Sequential(
            ResBlock3d(64, 128, downsample=True),
            ResBlock3d(128, 128, downsample=False),
        )

        #self.layer3 = nn.Sequential(
        #    ResBlock3d(128, 256, downsample=True),
        #    ResBlock3d(256, 256, downsample=False),
        #)

        #self.layer4 = nn.Sequential(
        #    ResBlock3d(256, 512, downsample=True),
        #    ResBlock3d(512, 512, downsample=False),
        #)

        self.gap = nn.AdaptiveAvgPool3d(1)
        #self.fc = nn.Conv3d(512, n_y_vals, 1)
        self.fc = nn.Conv3d(128, n_y_vals, 1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        x = self.gap(x)
        x = self.fc(x)
        x = flatten(x, start_dim=1)
        return x


def load_model_from_hub(model_name, model_path=None, pretrain=False):
    model = hub_load('pytorch/vision:v0.10.0', model_name,
                     pretrained=pretrain)
    if model_path:
        model.load_state_dict(load(model_path))
        model.to(double)
        model.eval()
    return model


def load_model(model_fn_name, classes, model_path=None):
    model = model_fn_name(classes)
    if model_path:
        model.load_state_dict(load(model_path))
        model.to(double)
        model.eval()
    return model
