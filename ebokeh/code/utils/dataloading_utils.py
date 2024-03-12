import numpy as np

from PIL import Image
from torch.nn import MSELoss, L1Loss, Module
from torchvision.transforms.functional import pad as pad_torch


def get_loss(loss=None, loss_args=None) -> Module:
    """
    Get the loss function.
    Args:
        loss: str, one of 'MSELoss', 'L1Loss', 'SmoothL1Loss', 'PSNRLoss'
        loss_args: dict, arguments for the loss function

    Returns:
        nn.Module, the loss function

    """
    loss_args = loss_args or {}

    if loss == 'MSELoss':
        loss = MSELoss(**loss_args)
    elif loss == 'L1Loss':
        loss = L1Loss(**loss_args)
    else:
        raise NotImplementedError(f"Loss {loss} not implemented yet.")

    return loss


def pad_to_divisible_image(img: Image, divisor=16):
    h, w = img.size
    padding = get_divisible_padding(h, w, divisor)
    img = np.asarray(img)
    img_padded = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)), mode='reflect')
    img_padded = Image.fromarray(img_padded)

    return img_padded


def center_pad(img, output_size):
    image_height, image_width = img.size
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad_torch(img, padding_ltrb, padding_mode='reflect')

    return img


def get_divisible_padding(h, w, divisor=16):
    """Get the padding size to make the input divisible."""
    l_pad = int(np.ceil((divisor - w % divisor) / 2)) if w % divisor != 0 else 0
    r_pad = int(np.floor((divisor - w % divisor) / 2)) if w % divisor != 0 else 0
    t_pad = int(np.ceil((divisor - h % divisor) / 2)) if h % divisor != 0 else 0
    b_pad = int(np.floor((divisor - h % divisor) / 2)) if h % divisor != 0 else 0
    return l_pad, r_pad, t_pad, b_pad
