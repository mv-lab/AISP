import torch
import numpy as np
from torchinfo import summary


def get_gmacs_and_params(model, device, input_size=(1, 3, 6, 1060, 1900), print_detailed_breakdown=False):
    """ This function calculates the total MACs and Parameters of a given pytorch model.

    Args:
        model: A pytorch model object
        input_size: (batch_size, num images, channels, height, width) - input dimensions for a single NTIRE test scene 

    Returns:
        total_mult_adds: The total number of GMacs for the given model and input size
        total_params: The total number of parameters in the model

    """
    model_summary = summary(model, device=device,input_size=input_size, verbose=2 if print_detailed_breakdown else 0)
    return model_summary.total_mult_adds/10**9, model_summary.total_params

def get_runtime(model, device, input_size=(1, 3, 6, 1060, 1900), num_reps=100):
    """ This function calculates the mean runtime of a given pytorch model.
    More info: https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/

    Args:
        model: A pytorch model object
        input_size: (batch_size, num images, channels, height, width) - input dimensions for a single NTIRE test scene
        num_reps: The number of repetitions over which to calculate the mean runtime

    Returns:
        mean_runtime: The everage runtime of the model over num_reps iterations

    """
    # Set measurement to device, in this case we set this to cuda
    #device = torch.device("cuda")
    model.to(device)
    # Define input, for this example we will use a random dummy input
    input = torch.randn(input_size, dtype=torch.float).to(device)
    # Define start and stop cuda events
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times=np.zeros((num_reps, 1))
    # Perform warm-up runs (that are normally slower)
    #with torch.no_grad():
    for _ in range(10):
        _ = model(input)
    # Measure actual runtime
    with torch.no_grad():
        for it in range(num_reps):
            starter.record()
            _ = model(input)
            ender.record()
            # Await for GPU to finish the job and sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            times[it] = curr_time / 1000 # Convert from miliseconds to seconds
    # Average all measured times
    mean_runtime = np.sum(times) / num_reps
    return mean_runtime