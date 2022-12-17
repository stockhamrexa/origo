'''
The SteganoGan module.
'''

import math
import os
import random
import time
import torch
import string
from imageio.v3 import imread, imwrite
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from tqdm import tqdm

from .critic import Critic
from .decoder import Decoder
from .encoder import Encoder
from .utils import bits_to_bytearray, bits_to_text, bytearray_to_text, text_to_bits, validate_file_type

_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_BLOCK_SIZE = 2000
_PAD_SIZE = 32

class SteganoGAN():
    '''
    The SteganoGan class consists of a critic network, an encoder network, and decoder network, which are used to embed
    text in images, and subsequently extract that text.
    '''

    def __init__(self, data_depth=1, model_type='custom', cuda=False, log_dir=None, verbose=False):
        if data_depth < 1 or data_depth > 8 or not isinstance(data_depth, int):
            raise ValueError(
                'The data_depth argument must be an integer between 1 and 8.'
            )

        if model_type not in ['custom', 'div2k', 'mscoco']:
            raise ValueError(
                'The model_type argument must be one of: custom, div2k, mscoco'
            )

        self.data_depth = data_depth
        self.model_type = model_type

        model_path = os.path.join(_ROOT_PATH, 'pretrained', model_type, str(data_depth))

        if not os.path.isdir(model_path):
            raise NotADirectoryError(
                'There is not a pretrained {} model with a data_depth of {}'.format(model_type, data_depth)
            )

        self.critic = Critic.load(os.path.join(model_path, 'critic.pt'))
        self.decoder = Decoder.load(os.path.join(model_path, 'decoder.pt'), data_depth=data_depth)
        self.encoder = Encoder.load(os.path.join(model_path, 'encoder.pt'), data_depth=data_depth)

        self.verbose = verbose
        self.set_device(cuda)
        self.log_dir = log_dir

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

            self.metrics_path = os.path.join(self.log_dir, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)

            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)

    def set_device(self, cuda=True):
        '''
        Sets the device depending on whether cuda is available or not.
        '''

        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')

        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device.')

            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device.')

            else:
                print('Using CUDA device.')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

    def set_critic(self, critic):
        '''
        Manually set the critic network. Must be a subclass of torch.nn.Module.
        '''

        if not issubclass(torch.nn.Module, type(critic)):
            raise ValueError(
                'The critic object must inherit from torch.nn.Module.'
            )

        self.critic = critic

    def set_decoder(self, decoder):
        '''
        Manually set the decoder network. Must be a subclass of torch.nn.Module.
        '''

        if not issubclass(torch.nn.Module, type(decoder)):
            raise ValueError(
                'The decoder object must inherit from torch.nn.Module.'
            )

        self.decoder = decoder

    def set_encoder(self, encoder):
        '''
        Manually set the encoder network. Must be a subclass of torch.nn.Module.
        '''

        if not issubclass(torch.nn.Module, type(encoder)):
            raise ValueError(
                'The encoder object must inherit from torch.nn.Module.'
            )

        self.encoder = encoder

    def _make_payload(self, width, height, text, pad=True, compress=True):
        '''
        Converts a string of text into a bit vector payload to be hidden in a cover image.

        :param width: Width of the cover image.
        :param height: Height of the cover image.
        :param text: The string of text to embed in the image.
        :param pad: Whether or not to use padding when repeating the payload.
        :param compress: Whether or not to compress the text when converting it to bits.
        :return: A (1, self.data_depth, height, width) tensor containing bit vector copies of text.
        '''

        message = text_to_bits(text, compress)

        if pad:
            message += [0] * _PAD_SIZE

        payload = message

        if len(payload) > width * height * self.data_depth:
            raise ValueError(
                'The size of the payload is {}, which is larger than the size of the image {}. Try again with a smaller payload or a larger image.'.format(len(payload), width * height * self.data_depth)
            )

        while len(payload) < width * height * self.data_depth:
            payload += message

        payload = payload[:width * height * self.data_depth]

        return torch.FloatTensor(payload).view(1, self.data_depth, width, height)

    def _find_payload(self, image, greedy=False, k=25, var_threshold=None):
        '''
        Finds the payload embedded within an image, if it exists.

        :param image: An (N, D, H, W) torch tensor that is the output of self.decoder.
        :param greedy: If true, will stop searching once the first index is found with a variance below var_threshold.
                       Requires var_threshold is not None. Default is False.
        :param k: An int that dictates how many indices should be returned. Default is 25.
        :param var_threshold: The variance threshold. Will ignore all results with a variance above this value if not
                              none. Default is None.
        :return: A string if the payload was found, else None.
        '''

        image = image.view(-1)
        image = torch.sigmoid(image)

        repeat_indices = self._find_repeats(image.data, greedy, k, var_threshold)

        image = image > 0.5
        bits = image.data.int().cpu().numpy().tolist()

        candidates = {}

        for idx in repeat_indices:
            for i in range(0, len(bits), idx.item()):
                decoded = bits_to_text(bits[i:i + idx.item()])

                if decoded != False and decoded != '':
                    if decoded in candidates:
                        candidates[decoded] += 1

                        if greedy:
                            return decoded

                    else:
                        candidates[decoded] = 1

        if len(candidates) > 0:
            return max(candidates, key=candidates.get)

        else: # Brute force check for solutions based on padding
            for candidate in bits_to_bytearray(bits).split(b'\x00' * (_PAD_SIZE // 8)):

                decoded = bytearray_to_text(bytearray(candidate))

                if decoded != False  and decoded != '':
                    if decoded in candidates:
                        candidates[decoded] += 1

                        if greedy:
                            return decoded

                    else:
                        candidates[decoded] = 1

            if len(candidates) > 0:
                return max(candidates, key=candidates.get)

            else:
                return None

    def _find_repeats(self, probs, greedy=False, k=2, var_threshold=None):
        '''
        Finds repeating patterns in a length n tensor by computing the average column-wise variance between each sub-array
        formed by folding the probs input at a given index and truncating any leftover elements.

        :param probs: A 1D tensor of length n, whose elements represent the probability of the digit at that index being
                      a 1.
        :param greedy: If true, will stop searching once the first index is found with a variance below var_threshold.
                       Requires var_threshold is not None. Default is False.
        :param k: An int that dictates how many indices should be returned. Default is 25.
        :param var_threshold: The variance threshold. Will ignore all results with a variance above this value if not
                              none. Default is None.
        :return: A length k tensor of indices.
        '''

        length = probs.shape[0]
        valid_indices = []

        for i in range(_BLOCK_SIZE, length // 2, _BLOCK_SIZE + 40): # Generate a list of all valid encoded block lengths
            for j in range(6):
                valid_indices.append(i + j * 8)
                valid_indices.append(i + j * 8 + 32)

        valid_indices = list(set(valid_indices))
        valid_indices = sorted(valid_indices)

        best_indices = torch.ones(len(valid_indices))

        for i in range(len(valid_indices)):
            idx = valid_indices[i]
            remainder = length % idx

            truncated_probs = probs[:length - remainder]
            folded = truncated_probs.view([length // idx, idx])
            var = torch.var(folded, dim=0) # Get the variance of each column

            avg_var = torch.sum(var).item() / idx

            if var_threshold:
                if avg_var <= var_threshold:
                    best_indices[i] = avg_var

                    if greedy:
                        return torch.tensor([idx]) # Stop when you hit the first index with a variance below var_threshold

            else:
                best_indices[i] = avg_var

        values, indices = torch.topk(best_indices, k=k, largest=False) # Get the k indices with the lowest variance

        if var_threshold:
            indices = indices[values <= var_threshold]

        return torch.tensor(valid_indices)[indices]

    def critique(self, image):
        '''
        Evaluate a single image using the critic network, where the image is identified by a file path.

        :param image: Path to the steganographic image to be critiqued.
        :return: The Wasserstein distance between the natural image distribution and the steganographic image.
        '''

        valid, _ = validate_file_type(image)

        if not valid:
            raise ValueError(
                'The image does not have a valid file extension. SteganoGAN only supports the PNG and JPEG formats.'
            )

        cover = imread(image) / 127.5 - 1.0  # Normalize the image to [-1, 1]

        if len(cover.shape) == 2: # If it is a single channel greyscale image
            cover = torch.FloatTensor(cover).unsqueeze(2)
            cover = cover.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

        else:
            cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover = cover.to(self.device)

        _, channels, width, height = cover.size()

        if channels == 4:  # If there is an alpha layer
            cover = cover[:, 0:3, :, :]

        score = torch.mean(self.critic(cover))

        if self.verbose:
            print('The Wasserstein distance between the natural image distribution and this image is: {}'.format(score.item()))

        return score.item()

    def encode(self, cover, output, text, pad=True, compress=True):
        '''
        Encode text in a single image, where the image is identified by a file path.

        :param cover: Path to the image to be used as a cover.
        :param output: Path where the generated image will be saved.
        :param text: Message to hide inside the image.
        :param pad: Whether or not to use padding when repeating the payload. Note: If padding is used, it will be
                    more difficult to decode the payload via brute force.
        :param compress: Whether or not to compress the text when converting it to bits. Note: If text is compressed,
                         you can fit a larger message into the image but it will be more difficult to decode the
                         payload via brute force.
        :return: A tensor representation of the image with text embedded in it.
        '''

        if not os.path.exists(cover):
            raise ValueError(
                'Unable to read {}.'.format(cover)
            )

        if not isinstance(output, str):
            raise ValueError(
                'The output parameter must be a string.'
            )

        if not isinstance(text, str):
            raise ValueError(
                'The text parameter must be a string.'
            )

        if not isinstance(pad, bool):
            raise ValueError(
                'The pad parameter must be a bool.'
            )

        if not isinstance(compress, bool):
            raise ValueError(
                'The compress parameter must be a bool.'
            )

        valid_cover, cover_type = validate_file_type(cover)
        valid_output, output_type = validate_file_type(output)

        if not valid_cover or not valid_output:
            raise ValueError(
                'The cover image or the output does not have a valid file extension. SteganoGAN only supports the PNG and JPEG formats.'
            )

        if cover_type != output_type:
            raise ValueError(
                'The cover image and the output image must have the same file extension.'
            )

        cover = imread(cover) / 127.5 - 1.0 # Normalize the image to [-1, 1]

        if len(cover.shape) == 2: # If it is a single channel greyscale image
            cover = torch.FloatTensor(cover).unsqueeze(2)
            cover = cover.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

        else:
            cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        _, channels, width, height = cover.size()

        if channels == 4: # If there is an alpha layer
            alpha_channel = cover[:, 3:, :, :][0]
            cover = cover[:, 0:3, :, :]

        payload = self._make_payload(width, height, text, pad, compress)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        if channels == 4: # Add the alpha layer back to the image
            generated = torch.cat([generated, alpha_channel], 0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')

        return torch.Tensor(generated)

    def decode(self, image, greedy=False, k=25, var_threshold=None):
        '''
        Decode text in a single image, where the image is identified by a file path.

        :param image: Path to the image to extract a message from.
        :param greedy: If true, will stop searching once the first index is found with a variance below var_threshold.
                       Requires var_threshold is not None. Default is False.
        :param k: An int that dictates how many indices should be returned. Default is 25.
        :param var_threshold: The variance threshold. Will ignore all results with a variance above this value if not
                              none. Default is None.
        :return: The raw output of self.decoder and the decoded payload, else None.
        '''

        if not os.path.exists(image):
            raise ValueError(
                'Unable to read {}.'.format(image)
            )

        if not isinstance(greedy, bool) or (greedy and not var_threshold):
            raise ValueError(
                'The parameter greedy must be a boolean, and can only be True if var_threshold is not None.'
            )

        if not isinstance(k, int):
            raise ValueError(
                'The parameter k must be an int.'
            )

        if var_threshold and (var_threshold < 0 or var_threshold > 1):
            raise ValueError(
                'The parameter var_threshold must be a number between 0 and 1.'
            )

        valid_type, _ = validate_file_type(image)

        if not valid_type:
            raise ValueError(
                'The image does not have a valid file extension. SteganoGAN only supports the PNG and JPEG formats.'
            )

        image = imread(image) / 255.0  # Normalize the image to [0, 1]

        if len(image.shape) == 2: # If it is a single channel greyscale image
            image = torch.FloatTensor(image).unsqueeze(2)
            image = image.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

        else:
            image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)

        image = image.to(self.device)

        _, channels, width, height = image.size()

        if channels == 4: # If there is an alpha layer
            image = image[:, 0:3, :, :]

        output = self.decoder(image)
        payload = self._find_payload(output, greedy=greedy, k=k, var_threshold=var_threshold)

        if self.verbose:
            if payload != None:
                print('The message is: {}'.format(payload))

            else:
                print('Failed to find message.')

        return output, payload

    def get_metrics(self, cover, num_iters=10, pad=True, compress=True, greedy=False, k=25, var_threshold=None):
        '''
        Generate random data to be encoded in a cover image and collect metrics on the SteganoGAN instance by performing
        the critique, encode, and decode operations, using the default arguments for those functions unless otherwise
        specified.

        :param cover: Path to the image to be used as a cover.
        :param num_iters: The number of iterations to average over. Default is 10.
        :param pad: Whether or not to use padding when repeating the payload. Note: If padding is used, it will be
                    more difficult to decode the payload via brute force.
        :param compress: Whether or not to compress the text when converting it to bits. Note: If text is compressed,
                         you can fit a larger message into the image but it will be more difficult to decode the
                         payload via brute force.
        :param greedy: If true, will stop searching once the first index is found with a variance below var_threshold.
               Requires var_threshold is not None. Default is False.
        :param k: An int that dictates how many indices should be returned. Default is 25.
        :param var_threshold: The variance threshold. Will ignore all results with a variance above this value if not
                              none. Default is None.
        :return: The average time in seconds to critique, decode, and encode random data, the average bits per pixel,
                 the average critique score, the average decoder accuracy, the number of times the message was decoded
                 successfully, the average peak to signal noise ratio, and the average structural similarity index
                 measure.
        '''

        if not os.path.exists(cover):
            raise ValueError(
                'Unable to read {}.'.format(cover)
            )

        valid, cover_type = validate_file_type(cover)

        if not valid:
            raise ValueError(
                'The cover image does not have a valid file extension. SteganoGAN only supports the PNG and JPEG formats.'
            )

        was_verbose = self.verbose
        self.verbose = False # Silence verbosity of function calls

        critique_time = 0
        decode_time = 0
        encode_time = 0

        bpp = 0
        psnr = 0
        ssim = 0

        critique_score = 0
        decoder_acc = 0
        successful_decode = 0

        loss = torch.nn.MSELoss()
        out_file = 'output.' + cover_type

        cover_img = imread(cover)

        if len(cover_img.shape) == 2:  # If it is a single channel greyscale image
            cover_img = torch.FloatTensor(cover_img).unsqueeze(2)
            cover_img = cover_img.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

        else:
            cover_img = torch.FloatTensor(cover_img).permute(2, 1, 0).unsqueeze(0)

        _, _, width, height = cover_img.size()
        payload_length = random.randint(1, math.floor((5 / (_BLOCK_SIZE + 40)) * self.data_depth * height * width)) # Pick a random payload length that is less than or equal to 1/8'th the size of the embeddable space in the image (8 bits to a byte)

        for _ in tqdm(range(num_iters), disable=not was_verbose):
            valid_chars = string.digits + string.ascii_letters + ' '
            payload = ''.join(random.choices(valid_chars, k=payload_length))

            encode_start = time.time()
            generated = self.encode(cover, output=out_file, text=payload, pad=pad, compress=compress)
            encode_time += time.time() - encode_start

            critique_start = time.time()
            critique_score += self.critique(image=out_file)
            critique_time += time.time() - critique_start

            decode_start = time.time()
            raw, decoded = self.decode(image=out_file, greedy=greedy, k=k, var_threshold=var_threshold)
            decode_time += time.time() - decode_start

            encoded_payload = self._make_payload(width, height, text=payload, pad=pad, compress=compress)
            accuracy = (raw >= 0.0).eq(encoded_payload >= 0.5).sum().float() / encoded_payload.numel()
            decoder_acc += accuracy.item()
            bpp += (self.data_depth * (2 * accuracy.item() - 1))
            generated = generated.permute(2, 1, 0).unsqueeze(0)
            psnr += (10 * torch.log10(4 / loss(generated / 255, cover_img / 255))).item()
            ssim += SSIM(generated, cover_img)

            if decoded == payload:
                successful_decode += 1

        os.remove(out_file)

        self.verbose = was_verbose

        critique_time /= num_iters
        decode_time /= num_iters
        encode_time /= num_iters
        bpp /= num_iters
        critique_score /= num_iters
        decoder_acc /= num_iters
        psnr /= num_iters
        ssim /= num_iters

        if self.verbose:
            print('Average Critique Time: {}\n'
                  'Average Decode Time: {}\n'
                  'Average Encode Time: {}\n'
                  'Average Bits Per Pixel: {}\n'
                  'Average Critique Score: {}\n'
                  'Average Decoder Accuracy: {}\n'
                  'Number Of Successful Decodes: {}/{}\n'
                  'Average Peak Signal To Noise Ratio: {}\n'
                  'Average Structural Similarity Index Measure: {}'.format(critique_time, decode_time, encode_time, bpp, critique_score, decoder_acc, successful_decode, num_iters, psnr, ssim))

        return critique_time, decode_time, encode_time, bpp, critique_score, decoder_acc, successful_decode, psnr, ssim

    def __str__(self):
        '''
        Represents the SteganoGAN class as a string.
        '''

        overview_string = '\tData Depth: {}\n' \
                          '\tModel Type: {}\n' \
                          '\tVerbose: {}\n' \
                          '\tCUDA: {}\n' \
                          '\tDevice: {}\n'.format(self.data_depth, self.model_type, self.verbose, self.cuda, self.device)

        if self.log_dir:
            dir_string = '\tLog Directory: {}\n' \
                         '\tMetrics Directory: {}\n' \
                         '\tSamples Directory: {}\n\n'.format(self.log_dir, self.metrics_path, self.samples_path)

        else:
            dir_string = '\tLog Directory: {}\n\n'.format(self.log_dir)

        critic = self.critic.__str__()
        decoder = self.decoder.__str__()
        encoder = self.encoder.__str__()

        model_str = 'Critic: {}\n\nDecoder: {}\n\nEncoder: {}'.format(critic, decoder, encoder)

        return 'SteganoGAN:\n{}{}{}'.format(overview_string, dir_string, model_str)