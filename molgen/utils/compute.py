
from molgen.utils.path import count_subfolders, get_most_recent_subfolder
import torch
import os

def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def load_run_folder(path):
    """
        Creates the run folder or handles existing checkpoints
        Arguments:
            path:str    path to run folder
            rank:int    process rank (parallelism)
        Returns:
            path:str    path to run folder or checkpoint to load
            load:bool   flag on whether to load from checkpoint or not
    """
    # Master rank makes the run folder
    if is_master_rank() and not os.path.exists(path):
        print("[-] Creating run folder...")
        os.mkdir(path) 

    # Look for a valid checkpoint folder
    if os.path.exists(path) and count_subfolders(path) > 0: 
        ckpt_path = get_most_recent_subfolder(path)
        return ckpt_path, True
    return path, False


def is_master_rank():
    try:
        RANK = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        return RANK == 0
    except: return True

def unique_list(x):
    return list(set(x))

def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)

def print_trainable_parameters(model):
    """
        Prints the number of trainable parameters in the model.
        Arguments:
            model(object): HF loaded LM model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )