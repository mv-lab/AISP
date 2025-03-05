import numpy as np
from PIL import Image as PILImage
import torch
import torch.distributions as dist


def add_heteroscedastic_gnoise(image, sigma_1_range=(5e-3, 5e-2), sigma_2_range=(1e-3, 1e-2)):
    """
    Adds heteroscedastic Gaussian noise to an image.
    
    Parameters:
    - image: PyTorch tensor of the image.
    - sigma_1_range: Tuple indicating the range of sigma_1 values.
    - sigma_2_range: Tuple indicating the range of sigma_2 values.
    
    Returns:
    - Noisy image: Image tensor with added heteroscedastic Gaussian noise.
    """
    # Randomly choose sigma_1 and sigma_2 within the specified ranges
    sigma_1 = torch.empty(image.size()).uniform_(*sigma_1_range)
    sigma_2 = torch.empty(image.size()).uniform_(*sigma_2_range)
    
    # Calculate the variance for each pixel
    variance = (sigma_1 ** 2) * image + (sigma_2 ** 2)
    
    # Generate the Gaussian noise
    noise = torch.normal(mean=0.0, std=variance.sqrt())
    
    # Add the noise to the original image
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

def add_gnoise(inp_img, var_low=5e-5, var_high=2e-4, variance=None):
    variance = var_low + np.random.rand() * (var_high - var_low)
    # Generate Gaussian noise with 0 mean and the specified variance
    noise = np.sqrt(variance) * torch.randn(inp_img.shape)
    return torch.clamp(inp_img + noise, 0., 1.)


#### Noise sampling based on UPI, CycleISP and Model-based Image Signal Processors

def real_noise_levels():
    """Generates random noise levels from a log-log linear distribution.
    This shot and read noise distribution covers a wide range of photographic scenarios.
    """
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.001]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10,log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.1]))
    read_noise = distribution.sample()
    line = lambda x: 1.34 * x + 0.22
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10,log_read_noise)
    return shot_noise, read_noise


def add_natural_noise(image):
    """Adds random shot (proportional to image) and read (independent) noise."""
    shot_noise, read_noise = real_noise_levels()

    assert torch.all(image) >= 0
    assert torch.all(shot_noise) >= 0
    assert torch.all(read_noise) >= 0
    #print (111, read_noise, shot_noise)
    variance = image * shot_noise + read_noise
    scale = torch.sqrt(variance)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), scale)
    noise = distribution.sample()
    noisy_raw = image + noise
    noisy_raw = torch.clamp(noisy_raw,0,1)
    return noisy_raw


if __name__ == "__main__":
    pass