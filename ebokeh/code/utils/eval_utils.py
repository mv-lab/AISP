import os

import numpy as np

from PIL import Image
from logging import warning, info, debug
from torch.cuda import Event
from torch.utils.data import DataLoader
from lpips import LPIPS

from png import Writer as PNGWriter

from code.data.dataset import BokehTransformDataset


def sanity_checks(gt_avail, calculate_metrics, output_dir, image_in_dir) -> None:
    if not gt_avail and calculate_metrics:
        raise ValueError("Cannot calculate metrics without ground truth images")

    if not gt_avail and output_dir is None:
        warning("No output is saved, this is probably not what you want, please set output_dir to a valid path")

    if not os.path.exists(image_in_dir):
        raise FileNotFoundError(f"Image directory {image_in_dir} does not exist")


def save_tensor_img(image, image_id, output_dir, output_format) -> None:
    debug(f"Saving image {image_id} as {output_format} to {output_dir}")
    if output_format == "png" and output_dir is not None:
        image = image.detach().cpu().numpy().transpose(1, 2, 0)*65535.
        image = image.astype(np.uint16)
        with open(os.path.join(output_dir, f"{image_id}.pred.png"), "wb") as f:
            w = PNGWriter(width=image.shape[1], height=image.shape[0], greyscale=False, bitdepth=16)
            img2list = image.reshape(-1, image.shape[1]*image.shape[2]).tolist()
            w.write(f, img2list)
    elif output_format == "jpg" and output_dir is not None:
        image = image.detach().cpu().numpy().transpose(1, 2, 0)*255.
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(os.path.join(output_dir, f"{image_id}.pred.jpg"))
    elif output_dir is not None:
        raise ValueError(f"Unsupported output format {output_format}")


def get_dataloader(image_in_dir, model, metadata_override=None, input_crop=None, gt_avail=False) -> DataLoader:
    return DataLoader(
        BokehTransformDataset(image_in_dir, metadata=metadata_override, input_crop=input_crop,
                              validation=True if gt_avail else False,
                              predict=True if not gt_avail else False,
                              include_lens_factor=any([any(model.enc_blks_apply_lens_factor),
                                                       model.middle_blk_apply_lens_factor,
                                                       any(model.dec_blks_apply_lens_factor)]),
                              include_pos_map=any([model.in_stage_use_pos_map,
                                                   any(model.enc_blks_use_pos_map),
                                                   model.middle_blks_use_pos_map,
                                                   any(model.dec_blks_use_pos_map),
                                                   model.out_stage_use_pos_map]),
                              ), batch_size=1)


def setup_timings(device) -> (Event, Event, list):
    info("Setting up cuda events for timing")
    warning("Timing inference will only work on GPU") if device == "cpu" else None
    return Event(enable_timing=True), Event(enable_timing=True), []


def setup_metrics(device) -> (LPIPS, list, list, list):
    return LPIPS(net='alex').to(device), [], [], []
