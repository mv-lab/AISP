from skimage.metrics import peak_signal_noise_ratio
from torch import Tensor, no_grad, mean


def calculate_lpips(lpips_alex, img0: Tensor, img1: Tensor):
    # NOTE: LPIPS expects image normalized to [-1, 1]
    img0 = 2 * img0 - 1.0
    img1 = 2 * img1 - 1.0

    with no_grad():
        distance = lpips_alex(img0, img1)

    if max(distance.shape) > 1:
        return mean(distance).item()
    else:
        return distance.item()


def calculate_psnr(img0, img1):
    return peak_signal_noise_ratio(img0, img1)
