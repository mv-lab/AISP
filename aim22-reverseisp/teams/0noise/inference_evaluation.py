# -*- coding:utf-8 _*-
import os
import torch
from options import BaseOptions
from torch.utils.data import DataLoader
from dataset import dataset
import logging
import models
from tqdm import tqdm
from utils import utils
from evaluation.performance import get_gmacs_and_params, get_runtime


def main():
    # settings
    args = BaseOptions().parse()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        device = torch.device('cuda')
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    # dataset and dataloader
    inference_dataset = dataset.Evaluation_Dataset(opt=args)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    # model architectures
    model = models.create_model(args)
    model.to(device)
    weights = os.path.join(args.checkpoints, f'{args.trained_model}.pth')
    model = utils.load_from_checkpoint(weights, model, device)
    output_folder = os.path.join(args.output_folder, args.trained_model)
    os.makedirs(output_folder, exist_ok=True)

    n_samples = len(inference_loader)
    with torch.no_grad():
        desc_phase = "Inference:" 
        tqbar = tqdm(inference_loader, leave=False, total=n_samples, desc=desc_phase)
        for batch_idx, batch_data in enumerate(tqbar, 0):

            name = batch_data['name'][0]
            img_output_path = os.path.join(output_folder, name)
            tqbar.set_description(name, refresh=True)

            input = batch_data['input'].to(device)
            estimation = model(input).clamp(0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

            dataset.save_raw(estimation, img_output_path)

    print("Running ops metrics")
    if args.trained_model == "p20":
        sample = (1, 3, 496,496)
    elif args.trained_model == "s7":
        sample = (1, 3, 504, 504)
    else:
        sample = (1, 3, 512, 512)

    with torch.no_grad():
        total_macs, total_params = get_gmacs_and_params(model, device, input_size=sample)
        mean_runtime = get_runtime(model, device, input_size=sample)


    print("runtime per image [s] : " + str(mean_runtime))
    print("number of operations [GMAcc] : " + str(total_macs))
    print("number of parameters  : " + str(total_params))

    metrics_path = os.path.join(output_folder, "readme.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Runtime per image {sample}[s] : " + str(mean_runtime))
        #f.write('\n')
        #f.write("number of operations [GMAcc] : " + str(total_macs))
        #f.write('\n')
        #f.write("number of parameters  : " + str(total_params))
        f.write('\n')
        f.write("CPU[1] / GPU[0] : 0")
        f.write('\n')
        f.write("Extra Data [1] / No Extra Data [0] : 0")
        f.write('\n')
        f.write("Other description: We have a Pytorch implementation, and report single GPU runtime. The method was trained on the training dataset (- 10% for validation) for 300 epochs.")


if __name__ == '__main__':
    main()
