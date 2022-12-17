'''
The critic module.
'''

import torch
from torch import nn

class Critic(nn.Module):
    '''
    The Critic module takes an image and predicts whether it is a cover
    image or a steganographic image.

    Input: (N, 3, H, W) image
    Output: (N, 1) probability
    '''

    def __init__(self, hidden_size=32):
        super().__init__()
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
        return nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, 1)
        )

    def forward(self, image):
        x = self._models(image)
        x = torch.mean(x.view(x.size(0), -1), dim=1)

        return x

    @classmethod
    def load(cls, path=None, device=None, **kwargs):
        '''
        Loads a pretrained Critic model from a given path.

        :param path: Path to the custom pretrained model. The state_dict must match the Critic architecture defined
                     by kwargs.
        :param device: A torch.cuda.device object.
        :param kwargs: An arbitrarily long dictionary of keyword args to be passed during Critic initialization.
        '''

        map_location = 'cpu' if device is None else device

        if path is None:
            raise ValueError(
                'You must provide a path to a pretrained models state_dict.'
            )

        critic = Critic(**kwargs)
        critic.load_state_dict(torch.load(path, map_location=map_location))

        return critic