import os
import json
import torch
import random
from transformers import set_seed

def set_train_seed(seed):
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    
def log_svd_shapes(iteration, U_dict, output_file):
    if os.path.exists(output_file):
        if os.stat(output_file).st_size == 0:
            all_results = {}
        else:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
    else:
        all_results = {}

    if str(iteration) not in all_results:
        all_results[str(iteration)] = {}

    for layer_name, matrix_data in U_dict.items():
        matrix_shape = matrix_data.shape

        all_results[str(iteration)][layer_name] = {
            "shape": matrix_shape 
        }

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)