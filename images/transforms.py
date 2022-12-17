'''
A collection of differentiable Kornia transformations.
'''

import kornia.augmentation as K
import numpy as np
import torch

from kornia.enhance import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, sharpness
from kornia.geometry.transform import center_crop, get_perspective_transform, resize, rotate, translate, warp_perspective

# Geometric transformations
class RandomCenterCrop(K.GeometricAugmentationBase2D):
    '''
    Performs a random center crop on a given image. Randomly crops the image to size [0, pct_x] * width, [0, pct_y] *
    height. Zero padding is added to maintain the original image size.
    '''

    def __init__(self, pct_x=0.1, pct_y=0.1, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomCenterCrop, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.pct_x = pct_x
        self.pct_y = pct_y
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        x_step_size = self.pct_x / num_buckets
        x_offset = self.pct_x % (x_step_size * (num_buckets - 1)) / 2
        x_range = np.arange(x_offset, self.pct_x, x_step_size)

        y_step_size = self.pct_y / num_buckets
        y_offset = self.pct_y % (y_step_size * (num_buckets - 1)) / 2
        y_range = np.arange(y_offset, self.pct_x, x_step_size)

        _, _, width, height = input.shape

        for x_size, y_size in zip(x_range, y_range):
            size = (int(width - x_size * width), int(height - y_size * height))

            yield center_crop(input, size)

    def compute_transformation(self, input, params, flags):
        return input

    def apply_transform(self, input, params, flags, transform=None):
        _, _, width, height = input.shape
        x_size = torch.FloatTensor([1]).uniform_(0.0, self.pct_x).item()
        y_size = torch.FloatTensor([1]).uniform_(0.0, self.pct_y).item()
        size = (int(width - x_size * width), int(height - y_size * height))

        return center_crop(input, size, padding_mode='zeros')

class RandomPerspective(K.GeometricAugmentationBase2D):
    '''
    Performs a random perspective warp on a given image. Randomly shifts each corner of the image inward by
    [min_distortion, max_distortion] * width in the x direction and [min_distortion, max_distortion] * height in the y
    direction. Zero padding is added to maintain the original image size.
    '''

    def __init__(self, min_distortion=0.0, max_distortion=0.05, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomPerspective, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_distortion = min_distortion
        self.max_distortion = max_distortion
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        _, _, width, height = input.shape
        start_points = torch.Tensor([[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
        pts_norm = torch.Tensor([[[1, 1], [-1, 1], [-1, -1], [1, -1]]])
        fx = torch.Tensor([width / 2])
        fy = torch.Tensor([height / 2])

        step_size = (self.max_distortion - self.min_distortion) / num_buckets
        offset = (self.max_distortion - self.min_distortion) % (step_size * (num_buckets - 1)) / 2

        for max_distortion in np.arange(self.min_distortion + offset, self.max_distortion, step_size):
            distortion = torch.FloatTensor(start_points.shape).uniform_(self.min_distortion, max_distortion)
            offset = distortion * pts_norm * torch.stack([fx, fy], dim=0).view(-1, 1, 2)
            end_points = start_points + offset

            M = get_perspective_transform(start_points, end_points)

            yield warp_perspective(input, M, dsize=(height, width))

    def compute_transformation(self, input, params, flags):
        return input

    def apply_transform(self, input, params, flags, transform=None):
        _, _, width, height = input.shape
        start_points = torch.Tensor([[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
        pts_norm = torch.Tensor([[[1, 1], [-1, 1], [-1, -1], [1, -1]]])
        fx = torch.Tensor([width / 2])
        fy = torch.Tensor([height / 2])

        distortion = torch.FloatTensor(start_points.shape).uniform_(self.min_distortion, self.max_distortion)
        offset = distortion * pts_norm * torch.stack([fx, fy], dim=0).view(-1, 1, 2)
        end_points = start_points + offset

        M = get_perspective_transform(start_points, end_points)

        return warp_perspective(input, M, dsize=(height, width))

class RandomResize(K.GeometricAugmentationBase2D):
    '''
    Performs a random resize on a given image. Randomly resizes the image to size width + [-pct_x, pct_x] * width,
    height + [-pct_y, pct_y] * height. Zero padding is added to maintain the original image size if the original image
    was shrunk, otherwise the image is cropped to maintain the original image size (Functionally zooming in).
    '''

    def __init__(self, pct_x=0.1, pct_y=0.1, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomResize, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.pct_x = pct_x
        self.pct_y = pct_y
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        x_step_size = (2 * self.pct_x) / num_buckets
        x_offset = (2 * self.pct_x) % (x_step_size * (num_buckets - 1)) / 2
        x_range = np.arange(-self.pct_x + x_offset, self.pct_x, x_step_size)

        y_step_size = (2 * self.pct_y) / num_buckets
        y_offset = (2 * self.pct_y) % (y_step_size * (num_buckets - 1)) / 2
        y_range = np.arange(-self.pct_y + y_offset, self.pct_x, x_step_size)

        _, _, width, height = input.shape

        for x_size, y_size in zip(x_range, y_range):
            size = (int(width + x_size * width), int(height + y_size * width))

            yield resize(input, size)

    def compute_transformation(self, input, params, flags):
        return input

    def apply_transform(self, input, params, flags, transform=None):
        _, _, width, height = input.shape
        x_size = torch.FloatTensor([1]).uniform_(-self.pct_x, self.pct_x).item()
        y_size = torch.FloatTensor([1]).uniform_(-self.pct_y, self.pct_y).item()
        size = (int(width + x_size * width), int(height + y_size * width))

        return resize(input, size)

class RandomRotate(K.GeometricAugmentationBase2D):
    '''
    Performs a random rotation on a given image. Randomly rotates the image [-degrees, degrees] around the center. Zero
    padding is added to maintain the original image size.
    '''

    def __init__(self, degrees=15.0, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomRotate, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_degrees = -degrees
        self.max_degrees = degrees
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        step_size = (self.max_degrees - self.min_degrees) / num_buckets
        offset = (self.max_degrees - self.min_degrees) % (step_size * (num_buckets - 1)) / 2

        for degrees in np.arange(self.min_degrees + offset, self.max_degrees, step_size):
            degrees = torch.FloatTensor([degrees])

            yield rotate(input, degrees)

    def compute_transformation(self, input, params, flags):
        return input

    def apply_transform(self, input, params, flags, transform=None):
        degrees = torch.FloatTensor([1]).uniform_(self.min_degrees, self.max_degrees)

        return rotate(input, degrees)

class RandomTranslate(K.GeometricAugmentationBase2D):
    '''
    Performs a random translation on a given image. Randomly shifts the image [-pct_x, pct_x] * width in the x direction
    and [-pct_y, pct_y] * height in the y direction. Zero padding is added to maintain the original image size.
    '''

    def __init__(self, pct_x=0.025, pct_y=0.025, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomTranslate, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.pct_x = pct_x
        self.pct_y = pct_y
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        x_step_size = (2 * self.pct_x) / num_buckets
        x_offset = (2 * self.pct_x) % (x_step_size * (num_buckets - 1)) / 2
        x_range = np.arange(-self.pct_x + x_offset, self.pct_x, x_step_size)

        y_step_size = (2 * self.pct_y) / num_buckets
        y_offset = (2 * self.pct_y) % (y_step_size * (num_buckets - 1)) / 2
        y_range = np.arange(-self.pct_y + y_offset, self.pct_x, x_step_size)

        _, _, width, height = input.shape

        for x_shift, y_shift in zip(x_range, y_range):
            shift = torch.cat((torch.FloatTensor([x_shift * width]), torch.FloatTensor([y_shift * height]))).unsqueeze(0)
            shift = torch.round(shift)

            yield translate(input, shift)

    def compute_transformation(self, input, params, flags):
        return input

    def apply_transform(self, input, params, flags, transform=None):
        _, _, width, height = input.shape
        x_shift = torch.FloatTensor([1]).uniform_(-self.pct_x, self.pct_x) * width
        y_shift = torch.FloatTensor([1]).uniform_(-self.pct_y, self.pct_y) * height
        shift = torch.cat((x_shift, y_shift)).unsqueeze(0)
        shift = torch.round(shift)

        return translate(input, shift)

# Intensity transformations
class RandomBrightness(K.IntensityAugmentationBase2D):
    '''
    Performs a random brightness adjustment on a given image. Randomly adjusts brightness by a factor of
    [min_brightness, max_brightness], where a brightness of 0 does not modify the image.
    '''

    def __init__(self, min_brightness=0.0, max_brightness=0.2, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomBrightness, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        step_size = (self.max_brightness - self.min_brightness) / num_buckets
        offset = (self.max_brightness - self.min_brightness) % (step_size * (num_buckets - 1)) / 2

        for factor in np.arange(self.min_brightness + offset, self.max_brightness, step_size):
            factor = torch.FloatTensor([factor])

            yield adjust_brightness(input, factor, clip_output=True)

    def apply_transform(self, input, params, flags, transform=None):
        factor = torch.FloatTensor([1]).uniform_(self.min_brightness, self.max_brightness)

        return adjust_brightness(input, factor, clip_output=True)

class RandomContrast(K.IntensityAugmentationBase2D):
    '''
    Performs a random contrast adjustment on a given image. Randomly adjusts contrast by a factor of [min_contrast,
    max_contrast], where a contrast of 1 does not modify the image.
    '''

    def __init__(self, min_contrast=0.8, max_contrast=1.0, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomContrast, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        step_size = (self.max_contrast - self.min_contrast) / num_buckets
        offset = (self.max_contrast - self.min_contrast) % (step_size * (num_buckets - 1)) / 2

        for factor in np.arange(self.min_contrast + offset, self.max_contrast, step_size):
            factor = torch.FloatTensor([factor])

            yield adjust_contrast(input, factor, clip_output=True)

    def apply_transform(self, input, params, flags, transform=None):
        factor = torch.FloatTensor([1]).uniform_(self.min_contrast, self.max_contrast)

        return adjust_contrast(input, factor, clip_output=True)

class RandomHue(K.IntensityAugmentationBase2D):
    '''
    Performs a random hue shift on a given image. Randomly shifts hue by a factor of [-hue, hue], where a hue of 0 does
    not modify the image and hue <= pi.
    '''

    def __init__(self, hue=0.2, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomHue, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_hue = -hue
        self.max_hue = hue
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        step_size = (self.max_hue - self.min_hue) / num_buckets
        offset = (self.max_hue - self.min_hue) % (step_size * (num_buckets - 1)) / 2

        for factor in np.arange(self.min_hue + offset, self.max_hue, step_size):
            factor = torch.FloatTensor([factor])

            yield adjust_hue(input, factor)

    def apply_transform(self, input, params, flags, transform=None):
        factor = torch.FloatTensor([1]).uniform_(self.min_hue, self.max_hue)

        return adjust_hue(input, factor)

class RandomSaturation(K.IntensityAugmentationBase2D):
    '''
    Performs a random saturation adjustment on a given image. Randomly adjusts saturation by a factor of
    [min_saturation, max_saturation], where a saturation of 1 does not modify the image.
    '''

    def __init__(self, min_saturation=0.8, max_saturation=1.1, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomSaturation, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        step_size = (self.max_saturation - self.min_saturation) / num_buckets
        offset = (self.max_saturation - self.min_saturation) % (step_size * (num_buckets - 1)) / 2

        for factor in np.arange(self.min_saturation + offset, self.max_saturation, step_size):
            factor = torch.FloatTensor([factor])

            yield adjust_saturation(input, factor)

    def apply_transform(self, input, params, flags, transform=None):
        factor = torch.FloatTensor([1]).uniform_(self.min_saturation, self.max_saturation)

        return adjust_saturation(input, factor)

class RandomSharpness(K.IntensityAugmentationBase2D):
    '''
    Performs a random sharpness adjustment on a given image. Randomly adjusts sharpness strength by a factor of
    [min_sharpness, max_sharpness], where a sharpness of 1 does not modify the image.
    '''

    def __init__(self, min_sharpness=0.8, max_sharpness=1.2, p=1.0, keepdim=True, same_on_batch=False):
        super(RandomSharpness, self).__init__(p=p, keepdim=keepdim, same_on_batch=same_on_batch)
        self.min_sharpness = min_sharpness
        self.max_sharpness = max_sharpness
        self.p = p
        self.keepdim = keepdim
        self.same_on_batch = same_on_batch

    def sample(self, input, num_buckets=5):
        step_size = (self.max_sharpness - self.min_sharpness) / num_buckets
        offset = (self.max_sharpness - self.min_sharpness) % (step_size * (num_buckets - 1)) / 2

        for factor in np.arange(self.min_sharpness + offset, self.max_sharpness, step_size):
            factor = torch.FloatTensor([factor])

            yield sharpness(input, factor)

    def apply_transform(self, input, params, flags, transform=None):
        factor = torch.FloatTensor([1]).uniform_(self.min_sharpness, self.max_sharpness)

        return sharpness(input, factor)

# Sequentially applies all of the geometric transformations
GEOMETRIC = K.AugmentationSequential(
    RandomRotate()
)

# Sequentially applies all of the intensity transformations
INTENSITY = K.AugmentationSequential(
    RandomBrightness(),
    RandomContrast(),
    RandomHue(),
    RandomSaturation(),
    RandomSharpness()
)

# Sequentially applies all of the transformations which can not be sampled
BINARY = K.AugmentationSequential(
    K.RandomGrayscale(p=0.1, keepdim=True),
    K.RandomHorizontalFlip(p=0.1, keepdim=True),
    K.RandomVerticalFlip(p=0.1, keepdim=True)
)