import importlib
import torch
from torch import nn
import logging
ACTIVATIONS = ["relu", "leaky", "mish", "tanh", "silu", "sigmoid", "gelu"]
MODELS = [
    "base", "base_m", "base_cm", 
    "medium_cm", "micro","base3bmix","base3bmix2",
    "base9_0cm", "base9_001","base9_001cm","base9_00x","base9_00xcm","unet9_00xcm","base9_001cm_mix","base9_001cm_mix2",
    "rgb2raw", "rfdn", "aimbase", "base2","base2mix", "base2bmix","base3","base3b","base3c", "base4", "upi", "base5", "base6", "base7",
    "base7_0","base7_01", "base7_02",  "base7_03", "base8", "base9","base9_0","base9_00","base10","base10_1", "base7_3x3", "base7_3x3f", "base7_3x3cm",
]

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_lib_name = "models." + model_name + "_model"
    modellib = importlib.import_module(model_lib_name)


    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        logging.error("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_lib_name, target_model_name))
        exit(0)

    return model

def create_model(opt, name=None):
    if name is None:
        name = opt.model
        
    model = find_model_using_name(name)
    instance = model(opt)
    logging.info("# Model [%s] created" % (type(instance).__name__))

    return instance



def activation(act_type="relu", slope=0.2):
    if act_type == "leaky":
        return nn.LeakyReLU(negative_slope=slope)
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "mish":
        return nn.Mish()
    elif act_type == "tanh":
        return nn.Tanh()
    elif act_type == "silu":
        return nn.SiLU()
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation [{act_type}]")