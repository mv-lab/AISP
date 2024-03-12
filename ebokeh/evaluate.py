import os

from tqdm import tqdm
from logging import info
from piq.ssim import ssim as calculate_ssim
from torch import no_grad
from torch.cuda import synchronize
from numpy import uint16

from code.utils.eval_utils import save_tensor_img, get_dataloader, setup_timings, setup_metrics, sanity_checks
from code.metrics.metrics import calculate_psnr, calculate_lpips
from code.model.ln_modules import DeBokehLn

"""
Beginning of configuration
"""

device = "cuda:0"  # Set to "cpu" if you want to run on CPU
mode = "Validation"  # Set to "Test" or "Validation" depending on which subset you want to evaluate on
image_in_dir = f"dataset/Bokeh Transformation/{mode}"  # Path to the image input directory
gt_avail = False  # Set to True if you have ground truth images in the input directory, required for metrics
artifact_dir = "modelzoo/EBokehNet.ckpt"  # Path to the model artifact
output_dir = f"out/Bokeh Transformation/{mode}"  # Put your output directory here, put None to disable output
output_format = "png"  # Set to "png" or "jpg"

time_inference = True  # Set to True to measure inference time, only works on GPU
calculate_metrics = False  # Set to True to calculate quantitative metrics

"""
End of configuration
"""

sanity_checks(gt_avail, calculate_metrics, output_dir, image_in_dir)

os.makedirs(output_dir, exist_ok=True) if output_dir is not None else None

info(f"Loading model from {artifact_dir}")
model = DeBokehLn.load_from_checkpoint(artifact_dir).to(device).eval()

info("Configuring DataLoader")
dataloader = get_dataloader(image_in_dir, model, gt_avail=gt_avail)

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

    if calculate_metrics:
        target = batch["target"].to(device)

        target_psnr = target[0].detach().cpu().numpy().transpose(1, 2, 0)*65535.
        target_psnr = target_psnr.astype(uint16)

        out_psnr = out[0].detach().cpu().numpy().transpose(1, 2, 0)*65535.
        out_psnr = out_psnr.astype(uint16)

        PSNRs.append(calculate_psnr(target_psnr, out_psnr))
        SSIMs.append(calculate_ssim(target, out).item())
        LPIPSs.append(calculate_lpips(alex, target[0], out[0]))

    if output_dir is not None:
        save_tensor_img(out[0], image_id, output_dir, output_format)

info("Inference complete")

print(f"Average inference time: {sum(timings)/len(timings)} ms") if time_inference else None

print(f"Average PSNR: {sum(PSNRs)/len(PSNRs)}") if calculate_metrics else None
print(f"Average SSIM: {sum(SSIMs)/len(SSIMs)}") if calculate_metrics else None
print(f"Average LPIPS: {sum(LPIPSs)/len(LPIPSs)}") if calculate_metrics else None

