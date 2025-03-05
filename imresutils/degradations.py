import numpy as np
import torch
import torch.nn.functional as F

from blur import apply_psf
from noise import add_natural_noise, add_heteroscedastic_gnoise


def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H,w]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """

    # Ensure the image is in [B, C, H, W] format for PyTorch operations
    # raw_image_4channel = raw.permute(2, 0, 1).unsqueeze(0)
    
    # Apply average pooling over a 2x2 window for each channel
    downsampled_image = F.avg_pool2d(raw, kernel_size=2, stride=2, padding=0)
    
    # Rearrange back to [H/4, W/4, 4] format
    downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0)
    
    return downsampled_image


def convert_to_tensor(image):
    """
    Checks if the input image is a numpy array or a tensor.
    If it's a numpy array, converts it to a tensor.
    
    Parameters:
    - image: The input image, can be either a numpy array or a tensor.
    
    Returns:
    - A PyTorch tensor of the image.
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to tensor
        image_tensor = torch.from_numpy(image.copy())
        # If the image is in HxWxC format, convert it to CxHxW format expected by PyTorch
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        # If it's already a tensor, just return it
        image_tensor = image
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")

    return image_tensor


def simple_deg_simulation(img, kernels, down=False):
    """
    Pipeline to add synthetic degradations to a RAW image.
    The pipeline is implemented in pytorch so it can be part of your Dataset class, and efficient!
    Possible Transformations:
    y = down(x * k) + n
    y = (x * k) + n
    y = (x * k)
    y = x + n
    """

    p_noise = np.random.rand()
    p_blur  = np.random.rand()

    img = convert_to_tensor(img)

    # Apply psf blur: (x * k)
    if p_blur > 0.5:
        img = apply_psf(img, kernels)

    # Apply downsampling down(x*k)
    if down:
        img = downsample_raw(img)
    
    # Add noise: down(x*k) + n | (x*k) + n | x + n
    if p_noise > 0.5:
        img = add_natural_noise(img)
    else:
        img = add_heteroscedastic_gnoise(img)
    
    img = torch.clamp(img, 0. , 1.)
    img = img.permute(1,2,0).detach().cpu().numpy()
    img = img.astype(np.float32)
    return img