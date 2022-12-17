'''
The encoder module.
'''

import torch
from torch import nn

class Encoder(nn.Module):
    '''
    The Encoder module takes a cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W) image, (N, D, H, W) data
    Output: (N, 3, H, W) steganographic image
    '''

    def __init__(self, data_depth=1, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_model()

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_model(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )

        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )

        self.conv3 = nn.Sequential(
            self._conv2d(2 * self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )

        self.conv4 = nn.Sequential(
            self._conv2d(3 * self.hidden_size + self.data_depth, 3)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image, data):
        x = self._models[0](image)
        x_list = [x]

        for layer in self._models[1:]:
            x = layer(torch.cat(x_list + [data], dim=1))
            x_list.append(x)

        x = image + x

        return x

    @classmethod
    def load(cls, path=None, device=None, **kwargs):
        '''
        Loads a pretrained Encoder model from a given path.

        :param path: Path to the custom pretrained model. The state_dict must match the Encoder architecture defined
                     by kwargs.
        :param device: A torch.cuda.device object.
        :param kwargs: An arbitrarily long dictionary of keyword args to be passed during Encoder initialization.
        '''

        map_location = 'cpu' if device is None else device

        if path is None:
            raise ValueError(
                'You must provide a path to a pretrained models state_dict.'
            )

        encoder = Encoder(**kwargs)
        encoder.load_state_dict(torch.load(path, map_location=map_location))

        return encoder