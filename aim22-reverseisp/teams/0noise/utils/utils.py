import torch
import  logging


def model_parameters(model):
    all_ = sum(p.numel() for p in model.parameters())
    trainable_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {all_:,} | Trainable params: {trainable_:,}"



def load_from_checkpoint(path, model, device):
    logging.info(f"# Loading {path} weights")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model
