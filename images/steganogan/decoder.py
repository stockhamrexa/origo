'''
The decoder module.
'''

import torch
from torch import nn

class Decoder(nn.Module):
    '''
    The Decoder module takes a steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W) steganographic image
    Output: (N, D, H, W) data
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
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv3 = nn.Sequential(
            self._conv2d(2 * self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv4 = nn.Sequential(
            self._conv2d(3 * self.hidden_size, self.data_depth)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image):
        x = self._models[0](image)
        x_list = [x]

        for layer in self._models[1:]:
            x = layer(torch.cat(x_list, dim=1))
            x_list.append(x)

        return x

    @classmethod
    def load(cls, path=None, device=None, **kwargs):
        '''
        Loads a pretrained Decoder model from a given path.

        :param path: Path to the custom pretrained model. The state_dict must match the Decoder architecture defined
                     by kwargs.
        :param device: A torch.cuda.device object.
        :param kwargs: An arbitrarily long dictionary of keyword args to be passed during Decoder initialization.
        '''

        map_location = 'cpu' if device is None else device

        if path is None:
            raise ValueError(
                'You must provide a path to a pretrained models state_dict.'
            )

        decoder = Decoder(**kwargs)
        decoder.load_state_dict(torch.load(path, map_location=map_location))

        return decoder