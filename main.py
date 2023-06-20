import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        super(Generator, self).__init__()

        layers = [nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
                  nn.ReLU(inplace=True)]

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.01)]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


def load_model(model_path=''):
    # Build model
    print('Build generator')
    generator = Generator()

    """Restore the trained generator and discriminator."""
    print('Load generator ...')
    generator.load_state_dict(torch.load(PATH_MODEL, map_location=lambda storage, loc: storage))
    return generator


def load_image(image_path, image_size=128, crop_size=178):
    image = Image.open(image_path)
    trans_comp = [transforms.Resize((image_size, image_size)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    transform = transforms.Compose(trans_comp)
    image = transform(image)
    return image


def encode_label(color):
    color_arr = {'Black': 1, 'Blond': 2, 'Brown': 3}
    label = []
    for c in color_arr:
        label.append(color == c)
    return torch.FloatTensor(label)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def translate_hair_color(model_path='', image_path=''):
    img_name = image_path.split('.')[0]
    # Load generator model
    model = load_model(model_path)

    with torch.no_grad():
        # load image
        image = load_image(image_path).to('cpu')
        source_image = denorm(image.permute((1, 2, 0)))
        image = image.view(1, image.size(0), image.size(1), image.size(2))

        # encode label
        black_encoded_color = encode_label('Black').to('cpu')
        black_encoded_color = black_encoded_color.view(1, -1)

        blond_encoded_color = encode_label('Blond').to('cpu')
        blond_encoded_color = blond_encoded_color.view(1, -1)

        brown_encoded_color = encode_label('Brown').to('cpu')
        brown_encoded_color = brown_encoded_color.view(1, -1)

        # translate color of hair
        black_output = denorm(model(image, black_encoded_color)).squeeze().permute((1, 2, 0))
        blond_output = denorm(model(image, blond_encoded_color)).squeeze().permute((1, 2, 0))
        brown_output = denorm(model(image, brown_encoded_color)).squeeze().permute((1, 2, 0))
        output = torch.cat((source_image, black_output, blond_output, brown_output), dim=1)

        # show results
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(output.numpy())
        # plt.show()

        # print(output.shape)
        # result_path = os.path.join('./results', '{}-translate.jpg'.format(img_name))
        # save_image(output, result_path, nrow=1, padding=0)
        # print('Saved real and fake images into {}...'.format(result_path))

        return output.numpy()


if __name__ == "__main__":
    PATH_MODEL = "./models/generator.ckpt"  # path to model
    PATH_IMAGE = "./images/demo.jpg"  # path to image
    result = translate_hair_color(PATH_MODEL, PATH_IMAGE)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(result)
    plt.show()
