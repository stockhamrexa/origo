'''
The Expectation Over Transform (EOT) module for applying adversarial watermarking to images.
'''

import os
import torch
from imageio.v3 import imread, imwrite
from tqdm import tqdm

from steganogan import SteganoGAN, int_to_string, validate_file_type
from transforms import BINARY, INTENSITY, GEOMETRIC

class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class EOT():
    '''
    TODO
    '''

    def __init__(self, model, id_size=32):
        super(EOT, self).__init__()

        if not isinstance(model, SteganoGAN):
            raise ValueError(
                'The model parameter must be an instance of the SteganoGAN object.'
            )

        if not isinstance(id_size, int):
            raise ValueError(
                'The id_size parameter must be an int.'
            )

        self.model = model
        self.id_size = id_size

        self.model.encoder.eval()
        self.model.decoder.eval()

    def _conv_encode(self, message):
        '''
        TODO

        :param message:
        :return:
        '''
        padding = torch.zeros([1, 4])
        message = torch.cat((padding, message), dim=1)

        conv1 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, bias=False)
        g1 = torch.Tensor([[[0, 1, 0, 1, 1]]])  # Generator matrix 1
        conv1.weight = torch.nn.Parameter(g1)

        conv2 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, bias=False)
        g2 = torch.Tensor([[[1, 0, 1, 1, 1]]])  # Generator matrix 2
        conv2.weight = torch.nn.Parameter(g2)

        p1 = (conv1(message) % 2)[0, :]  # First parity bit
        p2 = (conv2(message) % 2)[0, :]  # Second parity bit

        return torch.stack((p1, p2), dim=1).reshape(1, p1.shape[0] + p2.shape[0]).detach()

    def _conv_decode(self, parity_bits, paths=[(torch.IntTensor([0, 0, 0, 0, 0]), 0)]):
        '''
        TODO

        :param parity_bits:
        :param paths:
        :return:
        '''

        zero = torch.IntTensor([0])
        one = torch.IntTensor([1])

        if parity_bits.shape[1] == 0: # If there are no parity bits left to decode
            paths.sort(key=lambda x: x[1])
            best_path, best_path_metric = paths[0]

            return best_path[5:], best_path_metric

        else:
            next_paths = []
            delete_idx = set()

            for message, path_metric in paths: # Iterate through all traversed paths
                last_state = message[-5:]
                next_bit_0 = torch.cat((last_state, zero), dim=0)
                parity_0 = self._conv_encode(next_bit_0.unsqueeze(0))[0, -2:]
                branch_metric_0 = torch.sum(torch.square(parity_bits[0, 0:2] - parity_0))
                next_paths.append((torch.cat((message, zero), dim=0), path_metric + branch_metric_0.item()))

                next_bit_1 = torch.cat((last_state, one), dim=0)
                parity_1 = self._conv_encode(next_bit_1.unsqueeze(0))[0, -2:]
                branch_metric_1 = torch.sum(torch.square(parity_bits[0, 0:2] - parity_1))
                next_paths.append((torch.cat((message, one), dim=0), path_metric + branch_metric_1.item()))

            for i in range(len(next_paths)):
                for j in range(len(next_paths)):
                    if i != j and torch.equal(next_paths[i][0][-5:], next_paths[j][0][-5:]):
                        if next_paths[j][1] > next_paths[i][1]:
                            delete_idx.add(j)

            for i in range(len(delete_idx)):
                idx = delete_idx.pop() - i
                del next_paths[idx]

            return self._conv_decode(parity_bits[:, 2:], next_paths)

    def _get_avg_fold(self, x):
        '''
        TODO make folds of size self.id_size * 2 (number of parity bits). Explain sigmoid
        :param x:
        :return:
        '''

        fold_size = self.id_size * 2
        temp = torch.FloatTensor([0.9899603058741643])
        # TODO determine whether or not to use temp

        x = x.flatten()
        x = x[:(x.shape[0] - x.shape[0] % fold_size)] # Cut off excess so that x is divisible by fold_size
        x = x.reshape(x.shape[0] // fold_size, fold_size)
        x = torch.mean(x, dim=0) # Get the average value for each of the fold_size bits

        #x = x * temp
        return x.sigmoid()

    def _projected_gradient_descent(self, x, target, eps=8/255, lr=9e-3, num_epochs=0, num_samples=10, random_seed=False):
        '''
        TODO

        :param x:
        :param target:
        :param eps:
        :param lr:
        :param num_epochs:
        :param num_samples:
        :param random_seed:
        :return:
        '''

        avg_losses = []

        if random_seed:
            seed = torch.zeros_like(x).uniform_(-1, 1) * eps
            x_adv = (x + seed).clip(0, 1).requires_grad_(True).to(self.model.device)

        else:
            x_adv = x.clone().detach().requires_grad_(True).to(self.model.device)

        optimizer = torch.optim.AdamW(params=[x_adv], lr=lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25, eta_min=lr/1e2)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.09, total_steps=num_epochs, div_factor=10, final_div_factor=1e3, three_phase=True, verbose=True)
        # TODO pick a scheduler that works

        from transforms import RandomRotate

        for _ in tqdm(range(num_epochs), disable=self.model.verbose != True):
            avg_loss = torch.Tensor([0])

            for _ in range(num_samples):
                z = GEOMETRIC(BINARY(INTENSITY(x_adv)))
                #z = RandomRotate()(x_adv)
                prediction = self.model.decoder(z)
                prediction = self._get_avg_fold(prediction)
                avg_loss += torch.nn.functional.mse_loss(prediction, target) / num_samples
                #loss = DiceBCELoss()
                #avg_loss += loss(prediction, target) / num_samples

            avg_loss.backward()
            optimizer.step()
            scheduler.step()
            #print(scheduler.get_last_lr())
            optimizer.zero_grad()

            #optimizer = torch.optim.Adam(params=[x_adv], lr=lr + i * .0001)

            x_adv.data = torch.max(torch.min(x_adv.data, x + eps), x - eps)
            x_adv.data = x_adv.data.clamp(0, 1)

            avg_losses.append(avg_loss.item())
            print(avg_loss.item())

        if self.model.verbose:
            print('Average loss per epoch: ', avg_losses)

        print('Average loss per epoch: ', avg_losses)
        return x_adv.detach(), avg_losses

    def encode(self, cover, output, id, eps=8/255, lr=9e-3, num_epochs=0, num_samples=10, random_seed=False):
        '''
        TODO

        :param cover:
        :param output:
        :param id:
        :param eps:
        :param lr:
        :param num_epochs:
        :param num_samples:
        :param random_seed:
        :return:
        '''

        if not os.path.exists(cover):
            raise ValueError(
                'Unable to read {}.'.format(cover)
            )

        if not isinstance(output, str):
            raise ValueError(
                'The output parameter must be a string.'
            )

        if not isinstance(id, int):
            raise ValueError(
                'The id parameter must be an int.'
            )

        if not isinstance(eps, float):
            raise ValueError(
                'The eps parameter must be a float.'
            )

        if eps < 0 or eps > 1:
            raise ValueError(
                'The eps parameter must have a value between 0 and 1.'
            )

        if not isinstance(lr, float) and not isinstance(lr, int):
            raise ValueError(
                'The lr parameter must be a float or an int.'
            )

        if not isinstance(num_epochs, int):
            raise ValueError(
                'The num_epochs parameter must be an int.'
            )

        if not isinstance(num_samples, int):
            raise ValueError(
                'The num_samples parameter must be an int.'
            )

        if not isinstance(random_seed, bool):
            raise ValueError(
                'The random_seed parameter must be a boolean.'
            )

        valid_cover, cover_type = validate_file_type(cover)
        valid_output, output_type = validate_file_type(output)

        if not valid_cover or not valid_output:
            raise ValueError(
                'The cover image or the output does not have a valid file extension. EOT only supports the PNG and JPEG formats.'
            )

        if cover_type != output_type:
            raise ValueError(
                'The cover image and the output image must have the same file extension.'
            )

        cover = imread(cover)

        if len(cover.shape) == 2: # If it is a single channel greyscale image
            cover = torch.FloatTensor(cover).unsqueeze(2)
            cover = cover.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

        else:
            cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        _, channels, width, height = cover.size()

        if channels == 4: # If there is an alpha layer
            alpha_channel = cover[:, 3:, :, :][0]
            cover = cover[:, 0:3, :, :]

        id = torch.FloatTensor([[int(i) for i in int_to_string(id, self.id_size)]])
        parity_bits = self._conv_encode(id)

        if id.shape[1] > width * height * self.model.data_depth:
            raise ValueError(
                'The size of the payload is {}, which is larger than the size of the image {}. Try again with a smaller payload or a larger image.'.format(id.shape[1], width * height * self.model.data_depth)
            )

        if num_epochs == 0: # Embed the id using the SteganoGAN encoder.
            cover = cover / 127.5 - 1.0 # Normalize the image to [-1, 1]
            payload = parity_bits[0].repeat((width * height * self.model.data_depth) // (2 * self.id_size) + 1)
            payload = payload[:width * height * self.model.data_depth].view(1, self.model.data_depth, width, height)

            cover = cover.to(self.model.device)
            payload = payload.to(self.model.device)

            generated = self.model.encoder(cover, payload)[0].clamp(-1.0, 1.0)
            generated = (generated + 1) * 127.5

        else:
            cover = cover / 255
            payload = parity_bits[0]

            cover = cover.to(self.model.device)
            payload = payload.to(self.model.device)

            generated, avg_losses = self._projected_gradient_descent(cover, payload, eps, lr, num_epochs, num_samples, random_seed)
            generated = generated[0].clamp(-1.0, 1.0) * 255

        if channels == 4: # Add the alpha layer back to the image
            generated = torch.cat([generated, alpha_channel], 0)

        generated = generated.permute(2, 1, 0).detach().cpu().numpy()
        imwrite(output, generated.astype('uint8'))

        if self.model.verbose:
            print('Encoding completed.')

        return torch.Tensor(generated)

    def decode(self, image):
        '''
        TODO

        :param image:
        :return:
        '''

        if not os.path.exists(image):
            raise ValueError(
                'Unable to read {}.'.format(image)
            )

        valid_type, _ = validate_file_type(image)

        if not valid_type:
            raise ValueError(
                'The image does not have a valid file extension. SteganoGAN only supports the PNG and JPEG formats.'
            )

        image = imread(image) / 255.0  # Normalize the image to [0, 1]

        if len(image.shape) == 2:  # If it is a single channel greyscale image
            image = torch.FloatTensor(image).unsqueeze(2)
            image = image.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

        else:
            image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)

        image = image.to(self.model.device)

        _, channels, width, height = image.size()

        if channels == 4:  # If there is an alpha layer
            image = image[:, 0:3, :, :]

        raw_output = self.model.decoder(image)
        raw_output = self._get_avg_fold(raw_output)
        message, path_metric = self._conv_decode(raw_output.unsqueeze(0))
        message = [str(i.item()) for i in message]
        message = int(''.join(message), 2)

        if self.model.verbose:
            print('The message is {}'.format(message))

        return raw_output, message, path_metric












x = SteganoGAN(verbose=False)
eot = EOT(x)

picture = 'a.png'
lr = 9e-2
eps = 255/255
num_samples = 20
random_seed = False

eot.encode(cover=picture, output='out.png', id=365891234, eps=eps, lr=lr, random_seed=random_seed, num_samples=num_samples, num_epochs=25)
print(eot.decode(image='out.png'))

'''
AdamW, no scheduler, mse_loss, lr=9e-3, eps=20/255, num_samples=20, random_seed=True
'''




pictures = ['a.png', 'b.png', 'c.png', 'd.png']
lr = [9e-3]
eps = [10/255]
num_samples = [20]
random_seed = [True, False]

image = imread('a.png') / 255.0  # Normalize the image to [0, 1]
image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
image = image.to(eot.model.device)


for i in lr:
    for j in eps:
        for k in num_samples:
            for l in random_seed:

                num_correct = 0
                num_correct_path_metric = 0

                for m in pictures:
                    target = random.randint(0, eot.id_size)
                    eot.encode(cover=m, output='out.png', id=target, eps=j, lr=i, random_seed=l, num_samples=k, num_epochs=25)

                    for i in range(-20, 21):
                        degrees = torch.FloatTensor([i])

                        image = imread('out.png') / 255.0  # Normalize the image to [0, 1]

                        if len(image.shape) == 2:  # If it is a single channel greyscale image
                            image = torch.FloatTensor(image).unsqueeze(2)
                            image = image.repeat(1, 1, 3).permute(2, 1, 0).unsqueeze(0)

                        else:
                            image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)

                        image = image.to(eot.model.device)

                        _, channels, width, height = image.size()

                        if channels == 4:  # If there is an alpha layer
                            image = image[:, 0:3, :, :]

                        image = rotate(image, degrees)

                        raw_output = eot.model.decoder(image)
                        raw_output = eot._get_avg_fold(raw_output)
                        message, path_metric = eot._conv_decode(raw_output.unsqueeze(0))
                        message = [str(i.item()) for i in message]
                        message = int(''.join(message), 2)

                        if message == target:
                            num_correct += 1
                            num_correct_path_metric += path_metric

                print('Adam, mse, lr: ' + str(i) + ', eps: ' + str(j) + ', num_samples: ' + str(
                    k) + ', random seed: ' + str(l))

                if num_correct == 0:
                    print('Pct correct: ' + str(0) + ' avg path metric: N/A')

                else:
                    print('Pct correct: ' + str(num_correct / (41 * 4)) + ' avg path metric: ' + str(
                        num_correct_path_metric / num_correct))
                print()


# TODO add a function that checks images resilience to a variety of transformations

'''
### 1e-2###
Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: True
Pct correct: 0.036585365853658534 avg path metric: 10.7297008348008

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: False
Pct correct: 0.07926829268292683 avg path metric: 11.37861889658066

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: True
Pct correct: 0.2926829268292683 avg path metric: 9.730014608746083

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: False
Pct correct: 0.573170731707317 avg path metric: 10.043956393886239

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: True
Pct correct: 0.2865853658536585 avg path metric: 9.013835116031956

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: False
Pct correct: 0.3475609756097561 avg path metric: 8.748615848129255

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: True
Pct correct: 0.21341463414634146 avg path metric: 8.391140233405999

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: False
Pct correct: 0.32926829268292684 avg path metric: 8.355467061132744

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: True
Pct correct: 0.03048780487804878 avg path metric: 6.0278544969856735

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: False
Pct correct: 0.024390243902439025 avg path metric: 10.651644183089957

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: True
Pct correct: 0.042682926829268296 avg path metric: 12.38903894541519

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: False
Pct correct: 0.06097560975609756 avg path metric: 11.754380524158478

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: True ###
Pct correct: 0.4329268292682927 avg path metric: 12.133332454297744

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: False ###
Pct correct: 0.38414634146341464 avg path metric: 11.660777501111466

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: True
Pct correct: 0.04878048780487805 avg path metric: 13.040019791573286

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: False
Pct correct: 0.012195121951219513 avg path metric: 11.668958939611912

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: True
Pct correct: 0.34146341463414637 avg path metric: 11.47079370166674

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: False
Pct correct: 0.21951219512195122 avg path metric: 11.214465152523998

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: True ###
Pct correct: 0.42073170731707316 avg path metric: 10.868096211898154

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: False ###
Pct correct: 0.5182926829268293 avg path metric: 10.307602326856816


### 9e-3 ###
Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: True
Pct correct: 0.07926829268292683 avg path metric: 10.293451372247477

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: False
Pct correct: 0.07926829268292683 avg path metric: 11.885705695129358

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: True
Pct correct: 0.29878048780487804 avg path metric: 9.856721805636676

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: False
Pct correct: 0.3231707317073171 avg path metric: 9.761190898786738

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: True
Pct correct: 0.29878048780487804 avg path metric: 9.006686873505918

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: False
Pct correct: 0.2865853658536585 avg path metric: 9.343977408840301

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: True ###
Pct correct: 0.31097560975609756 avg path metric: 8.376919442824288

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: False ###
Pct correct: 0.4024390243902439 avg path metric: 8.419591265420118

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: False
Pct correct: 0.006097560975609756 avg path metric: 12.377961367368698

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: True
Pct correct: 0.04878048780487805 avg path metric: 12.128372199833393

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: False
Pct correct: 0.0975609756097561 avg path metric: 11.840830813162029

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: True ###
Pct correct: 0.35365853658536583 avg path metric: 11.941130898218473

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: False ###
Pct correct: 0.35365853658536583 avg path metric: 11.232131394140165

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: False
Pct correct: 0.03048780487804878 avg path metric: 11.70099522396922

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: True
Pct correct: 0.23170731707317074 avg path metric: 11.469554234467642

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: False
Pct correct: 0.3231707317073171 avg path metric: 11.713993184268475

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: True ###
Pct correct: 0.6402439024390244 avg path metric: 10.2064546055737

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: False ###
Pct correct: 0.45121951219512196 avg path metric: 10.454784134478384


### 1e-3 ###
Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: True
Pct correct: 0.024390243902439025 avg path metric: 11.831125125288963

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: False
Pct correct: 0.0975609756097561 avg path metric: 11.475813582772389

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: True
Pct correct: 0.2865853658536585 avg path metric: 9.67800475696617

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: False
Pct correct: 0.3475609756097561 avg path metric: 10.073458948809849

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: True
Pct correct: 0.2865853658536585 avg path metric: 9.31812192784979

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: False
Pct correct: 0.32926829268292684 avg path metric: 9.078749612111736

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: True ###
Pct correct: 0.29878048780487804 avg path metric: 8.533988917816659

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: False ###
Pct correct: 0.49390243902439024 avg path metric: 8.120017925438322

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: False
Pct correct: 0.024390243902439025 avg path metric: 12.184565711766481

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: True
Pct correct: 0.11585365853658537 avg path metric: 12.575324983090946

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: False
Pct correct: 0.1951219512195122 avg path metric: 12.624070153106004

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: True
Pct correct: 0.23780487804878048 avg path metric: 11.504670903946344

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: False
Pct correct: 0.2682926829268293 avg path metric: 11.540015049956061

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: True
Pct correct: 0.03048780487804878 avg path metric: 11.624376729130745

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: False
Pct correct: 0.018292682926829267 avg path metric: 12.325958547492823

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: True
Pct correct: 0.35365853658536583 avg path metric: 11.710849375022446

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: False
Pct correct: 0.32926829268292684 avg path metric: 10.72124058601481

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: True ###
Pct correct: 0.524390243902439 avg path metric: 10.541485070944006

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: False ###
Pct correct: 0.5853658536585366 avg path metric: 10.39017817182806


### 9e-2 ###
Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: True
Pct correct: 0.12804878048780488 avg path metric: 9.977989606204487

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: False
Pct correct: 0.10975609756097561 avg path metric: 11.593821823596954

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: True
Pct correct: 0.3231707317073171 avg path metric: 9.841265192747397

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: False
Pct correct: 0.27439024390243905 avg path metric: 9.853659301954838

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: True
Pct correct: 0.43902439024390244 avg path metric: 9.333417402839082

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: False
Pct correct: 0.2621951219512195 avg path metric: 8.576407697225033

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: True
Pct correct: 0.36585365853658536 avg path metric: 8.26262725610286

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: False
Pct correct: 0.27439024390243905 avg path metric: 8.14959525068601

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: True
Pct correct: 0.006097560975609756 avg path metric: 13.028328120708466

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: False
Pct correct: 0.006097560975609756 avg path metric: 11.377261959016323

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: True
Pct correct: 0.07317073170731707 avg path metric: 11.964243852222959

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: False
Pct correct: 0.11585365853658537 avg path metric: 12.554593320739897

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: True ###
Pct correct: 0.3780487804878049 avg path metric: 11.796154683155399

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: False ###
Pct correct: 0.3353658536585366 avg path metric: 11.459241521290757

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: True
Pct correct: 0.006097560975609756 avg path metric: 11.99931064248085

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: False
Pct correct: 0.042682926829268296 avg path metric: 11.579716237527984

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: True
Pct correct: 0.18902439024390244 avg path metric: 11.64716176099835

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: False
Pct correct: 0.22560975609756098 avg path metric: 11.266157866078052

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: True ###
Pct correct: 0.5609756097560976 avg path metric: 10.440607926869037

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: False ###
Pct correct: 0.5121951219512195 avg path metric: 10.286672041551876


### 9e-4 ###
Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: True
Pct correct: 0.06097560975609756 avg path metric: 10.477369979955256

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 1, random seed: False
Pct correct: 0.10365853658536585 avg path metric: 12.023746908806702

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: True
Pct correct: 0.3475609756097561 avg path metric: 9.949890592380575

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 5, random seed: False
Pct correct: 0.25609756097560976 avg path metric: 9.454037487506866

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: True
Pct correct: 0.3780487804878049 avg path metric: 9.28605599875652

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 10, random seed: False
Pct correct: 0.3353658536585366 avg path metric: 9.084778269109401

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: True ###
Pct correct: 0.4634146341463415 avg path metric: 8.435270651488713

Adam, mse, lr: 20, eps: 0.0196078431372549, num_samples: 20, random seed: False ###
Pct correct: 0.3170731707317073 avg path metric: 8.024887276025346

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: True
Pct correct: 0.006097560975609756 avg path metric: 9.657696517184377

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 5, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: True
Pct correct: 0.04878048780487805 avg path metric: 12.419132157228887

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 10, random seed: False
Pct correct: 0.09146341463414634 avg path metric: 12.63373668094476

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: True ###
Pct correct: 0.47560975609756095 avg path metric: 11.616173057793043

Adam, mse, lr: 20, eps: 0.03137254901960784, num_samples: 20, random seed: False ###
Pct correct: 0.3475609756097561 avg path metric: 11.623464891392933

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: True
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 1, random seed: False
Pct correct: 0 avg path metric: N/A

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: True
Pct correct: 0.018292682926829267 avg path metric: 13.403559674819311

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 5, random seed: False
Pct correct: 0.16463414634146342 avg path metric: 11.8965629460635

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: True
Pct correct: 0.3231707317073171 avg path metric: 11.61841238137194

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 10, random seed: False
Pct correct: 0.3048780487804878 avg path metric: 11.137455507516862

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: True ###
Pct correct: 0.7073170731707317 avg path metric: 10.472306318870135

Adam, mse, lr: 20, eps: 0.0392156862745098, num_samples: 20, random seed: False ###
Pct correct: 0.524390243902439 avg path metric: 10.084292284792376
'''

'''
To-Do:
    - Pick the best loss function
    - Pick the best optimizer
    - Pick the best scheduler
    - Tune hyperparameters 
    - Implement a function to test the robustness of an encoded image to each of those transforms
    - Document
    - CLI
    
SteganoGAN:
    Update CLI if necessary
    Pick a number to encode based on image size and numbers that have not been chosen yet (no palindromes)
    
TextGAN:
    Hash function for a string of text
    Validate characters in the string
    Optionally convert to an image and pass to steganogan
    Make a command line interface

Origo:
    Make a requirements.txt file
    Package everything as a command line tool
    Make a setup command that either builds website or command line tool or both
    Build a website
    Perform file validation
    Build an API
'''