import os

import torchvision.transforms as T

from tqdm import tqdm
from logging import info
from piq.ssim import ssim as calculate_ssim
from torch import no_grad
from torch.cuda import synchronize

from code.utils.eval_utils import save_tensor_img, get_dataloader, setup_timings, setup_metrics, sanity_checks
from code.metrics.metrics import calculate_psnr, calculate_lpips
from code.model.ln_modules import DeBokehLn

"""
Beginning of configuration
"""

device = "cuda:0"  # Set to "cpu" if you want to run on CPU
image_in_dir = "dataset/EBB_Val294/Validation"  # Path to the image input directory
gt_avail = False  # Set to True if you have ground truth images in the input directory, required for metrics
artifact_dir = "modelzoo/EBokehNet_s_EBB_best.ckpt"  # Path to the model artifact
output_dir = "out/EBB_Val294/Validation"  # Put your output directory here, put None to disable output
output_format = "jpg"  # Set to "png" or "jpg"
input_crop = (1024, 1440)  # If higher than input resolution, will perform reflection pad instead
metadata_override = {"lens_factor": 1., "bokeh_strength": 0.7}  # Useful for datasets without metadata

time_inference = True  # Set to True to measure inference time, only works on GPU
calculate_metrics = True  # Set to True to calculate PSNR only works if GT images are already in the output directory

"""
End of configuration
"""

sanity_checks(gt_avail, calculate_metrics, output_dir, image_in_dir)

os.makedirs(output_dir, exist_ok=True) if output_dir is not None else None

info(f"Loading model from {artifact_dir}")
model = DeBokehLn.load_from_checkpoint(artifact_dir).to(device).eval()

info("Configuring DataLoader")
dataloader = get_dataloader(image_in_dir, model, metadata_override, input_crop)

starter, ender, timings = setup_timings(device) if time_inference else (None, None, None)
alex, PSNRs, SSIMs, LPIPSs = setup_metrics(device) if calculate_metrics else (None, None, None, None)

info("Starting inference")

for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    source = batch["source"].to(device)
    lens_factor = batch["lens_factor"].view(batch["lens_factor"].shape[0], 1, 1, 1).to(device)
    bokeh_strength = batch["bokeh_strength"].to(device)
    pos_map = batch["pos_map"].to(device)
    image_id = batch["image_id"][0][0]

    with no_grad():
        starter.record() if time_inference else None

        out = model.model(source, bokeh_strength, lens_factor, pos_map)

        ender.record() if time_inference else None
        synchronize() if time_inference else None
        timings.append(starter.elapsed_time(ender)) if time_inference else None

    if input_crop is not None:
        out = T.CenterCrop((batch["resolution"][0][1].item(), batch["resolution"][0][0].item()))(out)

    if calculate_metrics:
        target = batch["target"].to(device)

        if input_crop is not None:
            target = T.CenterCrop((int(batch["resolution"][0][1].item()), int(batch["resolution"][0][0].item())))(target)

        PSNRs.append(calculate_psnr(target[0].detach().cpu().numpy(), out[0].detach().cpu().numpy()))
        SSIMs.append(calculate_ssim(target, out).item())
        LPIPSs.append(calculate_lpips(alex, target[0], out[0]))

    if output_dir is not None:
        save_tensor_img(out[0], image_id, output_dir, output_format)

info("Inference complete")

print(f"Average inference time: {sum(timings)/len(timings)} ms") if time_inference else None

print(f"Average PSNR: {sum(PSNRs)/len(PSNRs)}") if calculate_metrics else None
print(f"Average SSIM: {sum(SSIMs)/len(SSIMs)}") if calculate_metrics else None
print(f"Average LPIPS: {sum(LPIPSs)/len(LPIPSs)}") if calculate_metrics else None

