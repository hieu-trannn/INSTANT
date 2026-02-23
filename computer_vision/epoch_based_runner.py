# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
import copy
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
import gc
import os
import json
import math


"""Hook installation"""

# Initialize dictionaries to store input_sum and gradients for each layer name
layer_inputs = {}  # {layer_name: [input_tensor1, input_tensor2, ...]}
layer_gradients = {}  # {layer_name: [grad_tensor1, grad_tensor2, ...]}
layer_weights = {} 
hook_handles_input = []  # To store the handles of the save_input_hook
hook_handles_grad = []  # To store the handles of the save_grad_hook
layer_modules = {}
U_dict = {}
U_grad_dict = {}

# data_json = {}

# Function to get layers which are compressed
# Get name of layers which need compression
def get_svd_layers(model):
    modules_compressed=[]
    for name, module in model.named_modules():
        if any(param.requires_grad for param in module.parameters()):
            if isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)):
                # Skip layers with "classifier" and "embeddings" in their name
                if "classifier" in name or "head.head" in name or "bert.pooler" in name:
                    continue
                modules_compressed.append(name)
    return modules_compressed

# Forward hook to store covariance of X for each layer
# def save_input_hook(name, module, input, output):
def save_input_hook(name, module, input, output):
    input = input[0].detach()
    # Reshape tensor based on its dimensions (3D)
    if input.dim() == 3: 
        if input.shape[2] < input.shape[1]:
            X_XT = torch.matmul( input.transpose(1, 2), input)
            average_X_XT = torch.mean(X_XT, dim=0)
        else:
            X_XT =  torch.matmul( input,input.transpose(1, 2))
            average_X_XT = torch.mean(X_XT, dim=0)
    elif input.dim() == 4:
        concatenated_tensor = input.view(input.shape[0], input.shape[1], -1)
        if concatenated_tensor.shape[2] < concatenated_tensor.shape[1]:
            X_XT =  torch.matmul(concatenated_tensor.transpose(1, 2), concatenated_tensor)
            average_X_XT = torch.mean(X_XT, dim=0)
        else:
            
            X_XT =  torch.matmul(concatenated_tensor, concatenated_tensor.transpose(1, 2))
            average_X_XT = torch.mean(X_XT, dim=0)
    else:
        raise ValueError(
            f"Unexpected tensor dimensions for layer {name}: "
            f"got {input.dim()}, expected 3"
        )

    if name not in layer_inputs:
        layer_inputs[name] = average_X_XT
    else:
        layer_inputs[name] += average_X_XT

# Backward hook to store gradient of Y for each layer
# def save_grad_hook(name, module, grad_input, grad_output):
def save_grad_hook(name, module, grad_input, grad_output):
    # global data_json
    # data_json[name] = grad_output[0].tolist()
    if grad_output[0].dim() == 3: 
        if grad_output[0].shape[2] < grad_output[0].shape[1]:
            gradX_gradXT =  torch.matmul(grad_output[0].transpose(1, 2), grad_output[0]) #[64,128,128]
            average_gradX_gradXT = torch.mean(gradX_gradXT, dim=0)
        # W = module.weight.data
        # gradXT_gradX =  torch.matmul(grad_output[0].transpose(1, 2), grad_output[0]) #[64,128,128]
        # average_gradXT_gradX = torch.mean(gradXT_gradX, dim=0)
        else:
            # W_WT = torch.matmul(W, W.transpose(0,1))
            # average_sum = torch.matmul(average_gradXT_gradX,W_WT) + torch.matmul(W_WT,average_gradXT_gradX)
            gradX_gradXT =  torch.matmul(grad_output[0],grad_output[0].transpose(1, 2)) #[64,128,128]
            average_gradX_gradXT = torch.mean(gradX_gradXT, dim=0)
    elif grad_output[0].dim() == 4:
        concatenated_tensor = grad_output[0].view(grad_output[0].shape[0], grad_output[0].shape[1], -1)
        if concatenated_tensor.shape[2] < concatenated_tensor.shape[1]:
            gradX_gradXT =  torch.matmul(concatenated_tensor.transpose(1, 2), concatenated_tensor)
            
            average_gradX_gradXT = torch.mean(gradX_gradXT, dim=0)
        else: 
            # gradX_gradXT =  torch.matmul(concatenated_tensor.transpose(1, 2), concatenated_tensor)
            gradX_gradXT =  torch.matmul(concatenated_tensor, concatenated_tensor.transpose(1, 2))
            average_gradX_gradXT = torch.mean(gradX_gradXT, dim=0)
    else:
        raise ValueError(
            f"Unexpected tensor dimensions for layer {name}: "
            f"got {grad_output[0].dim()}, expected 3"
        )
    # Initialize the list if it doesn't exist for this layer
    if name not in layer_gradients:
        layer_gradients[name] = average_gradX_gradXT
        # layer_gradients[name] = average_gradXT_gradX
        # layer_gradients[name] = average_sum
    else:
        layer_gradients[name] += average_gradX_gradXT
        # layer_gradients[name] += average_gradXT_gradX
        # layer_gradients[name] += average_sum
    del gradX_gradXT, average_gradX_gradXT
    # del gradXT_gradX, average_gradX_gradXT
    # del gradXT_gradX, average_gradXT_gradX, W_WT, average_sum
#     torch.cuda.empty_cache()
#     gc.collect()
# def add_activate_to_module():
    # Monkey-patch nn.Module to initialize activate
    if not hasattr(nn.Module, '_original_init'):
        nn.Module._original_init = nn.Module.__init__
        
        def new_init(self, *args, **kwargs):
            self._original_init(*args, **kwargs)
            # Initialize activate as a boolean attribute
            if not hasattr(self, 'activating'):
                self._parameters.pop('activating', None)  # Ensure it's not a parameter
                self._buffers.pop('activating', None)    # Ensure it's not a buffer
                self.__dict__['activating'] = True       # Set directly in __dict__
        
        nn.Module.__init__ = new_init

    # Add enable_compression method
    def enable_compression(self):
        self.__dict__['activating'] = True  # Set directly to avoid __setattr__
        for module in self.children():
            module.enable_compression()
        return self
    nn.Module.enable_compression = enable_compression

    # Add disable_compression method
    def disable_compression(self):
        self.__dict__['activating'] = False  # Set directly to avoid __setattr__
        for module in self.children():
            module.disable_compression()
        return self
    nn.Module.disable_compression = disable_compression

    # Initialize activate for existing modules
    def initialize_activate(self):
        self.__dict__['activating'] = True
        for module in self.children():
            module.initialize_activate()
    nn.Module.initialize_activate = initialize_activate
    
# REGISTER HOOK FOR LINEAR LAYER
def register_hooks(model):
    global hook_handles_input, hook_handles_grad, hook_handles_weight  # Modify the global lists of hook handles
    # print("Registering hooks for specific layers...")

    for name, module in model.named_modules():
        if any(param.requires_grad for param in module.parameters()):
            if isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)) :
                # Skip layers with "classifier" and "embeddings" in their name
                if "classifier" in name or "head." in name or "cpb_mlp" in name or "downsample" in name :
                    continue

                # print(f"Registering hooks for: {name} (Linear)")
                # handle_forward = module.register_forward_hook(lambda module, input, output, name=name: save_input_hook(name, input))
                # handle_backward = module.register_backward_hook(lambda module, grad_input, grad_output, name=name: save_grad_hook(name, grad_output))
                
                handle_forward = module.register_forward_hook(lambda module, input, output, name=name: save_input_hook(name, module, input, output))
                handle_backward = module.register_backward_hook(lambda module, grad_input, grad_output, name=name: save_grad_hook(name, module, grad_input, grad_output))

                hook_handles_input.append(handle_forward)
                hook_handles_grad.append(handle_backward)

# Unregister only the specific hooks
def unregister_hooks():
    global hook_handles_input, hook_handles_grad  # Access the global list of hook handles

    for handle in hook_handles_input:
        handle.remove()  # Unregister the input hook
    
    for handle in hook_handles_grad:
        handle.remove()  # Unregister the gradient hook

    hook_handles_input = []  # Clear the list of input hook handles
    hook_handles_grad = []  # Clear the list of gradient hook handles

    # Clear the layer input_sum and gradients dictionaries
    layer_inputs.clear()
    layer_gradients.clear()
    U_dict.clear()
    U_grad_dict.clear()
    # layer_weights.clear()  # Clear the dictionary storing weights
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()  # Release CUDA cache

# SVD calculation

def SVD_expected_value(input, var = 0.95, p = 0):
    U, S, V = torch.svd(input)

    # Calculate the total energy (sum of squared singular values)
    total_energy = torch.sum(S**2)

    # Compute the cumulative energy
    cumulative_energy = torch.cumsum(S**2, dim=0)

    # Find the number of singular values needed to retain 95% of the total energy
    energy_threshold = var * total_energy
    num_singular_values = torch.sum(cumulative_energy <= energy_threshold)
    num_singular_values = num_singular_values + p
    if num_singular_values >= U.shape[1]:
        num_singular_values = U.shape[1]-1
        
    if num_singular_values+1 > 0 and num_singular_values+1 < U.shape[1] :
        retained_energy = cumulative_energy[num_singular_values+1] 
        U_truncated = U[:, :num_singular_values+1]
        # U_truncated = U[:num_singular_values+1, :]
    elif num_singular_values+1 >= U.shape[1]:
        retained_energy = total_energy
        U_truncated = U[:, :num_singular_values]
        # U_truncated = U[:num_singular_values+1, :]
    else:
        retained_energy = 0 
    retained_energy_per = retained_energy/total_energy

    # Energy offset
    U_truncated = U_truncated * math.sqrt(1/retained_energy_per)
    return U_truncated

# SVD calculation
def SVD_projection(input, ratio =2):
    U, S, V = torch.svd(input)

    num_singular_values = U.shape[1] // ratio
    U_truncated = U[:, :num_singular_values]

    return U_truncated

# random projection scheme
def random_gaussian_projector(input, ratio=4):
    B = input.shape[0]
    r = B // ratio
    device = input.device
    dtype = input.dtype
    
    P = torch.randn(B, r, device=device, dtype=dtype) * (1/r) ** 0.5
    return P

def log_svd_shapes(iteration, U_dict, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Check if the file is empty
        if os.stat(output_file).st_size == 0:
            all_results = {}  # If the file is empty, start with an empty dictionary
        else:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
    else:
        all_results = {}  # If the file doesn't exist, start with an empty dictionary
    # # Initialize the structure for the current epoch and iteration if not exists
    # if str(epoch) not in all_results:
    #     all_results[str(epoch)] = {}

    if str(iteration) not in all_results:
        all_results[str(iteration)] = {}

    # For each layer in U_dict, log its shape (only the shape, not other data)
    for layer_name, matrix_data in U_dict.items():
        # matrix_data[0] contains the matrix
        matrix_shape = matrix_data[0].shape  # Get the shape of the matrix

        # Save only the shape of the matrix for each layer
        all_results[str(iteration)][layer_name] = {
            "shape": matrix_shape  # Only save the shape
        }

    # Write the updated results back to the file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
def parse_variables(input_str):
    # Initialize variables with default values
    compress = False
    compress_random = False
    compress_espace = False
    over_sam = 0
    var = 0.0
    calib_iter = 0
    compression_iter = 0
    ratio = 1  # or 0 or None depending on your logic
    
    # Split input string into lines and process each
    for line in input_str.strip().split('\n'):
        # Remove whitespace and skip empty lines
        line = line.strip()
        if not line:
            continue
            
        # Split on '=' and clean up parts
        try:
            key, value = [part.strip() for part in line.split('=')]
            
            # Assign values with correct types
            if key == 'compress':
                compress = value.lower() == 'true'
            elif key == 'over_sam':
                over_sam = int(value)
            elif key == 'var':
                var = float(value)
            elif key == 'calib_iter':
                calib_iter = int(value)
            elif key == 'compression_iter':
                compression_iter = int(value)
            # update ratio, self.compress_random,self.compress_espace like compress 
            elif key == 'ratio':
                ratio = int(value)
            if key == 'compress_random':
                compress_random = value.lower() == 'true'
            if key == 'compress_espace':
                compress_espace = value.lower() == 'true'
            
                
        except (ValueError, AttributeError):
            continue
            
    return compress, compress_random, compress_espace, over_sam, var, calib_iter, compression_iter, ratio

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_forward = 0
        self.total_backward = 0
        self.total_calibration = 0
        self.total_backward_optimizer = 0

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
            
        # print("Meta keys: " ,self.meta["config"])
        cfg = self.meta["config"]
        
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        
        
        time.sleep(2)  # Prevent possible deadlock during epoch transitio
        # print("After Calibration Phase:")
        #Training
        # data_iter = iter(self.data_loader)

        dataset_name = self.data_loader.dataset.__class__.__name__

        # print("Config: ", cfg)
        # calib_iter = cfg[calib_iter]
        # calib_steps = list(range(calib_iter))
        # var = cfg[var]
        # p = cfg[over_sam]
        # compression_iter = cfg[compression_iter]
        compress, compress_random, compress_espace, p, var, calib_iter, compression_iter, ratio = parse_variables(cfg)

        calib_steps = list(range(calib_iter))
        print(f"Config data: INSTANT compress: {compress}, INSTANT_random: {compress_random}, ESPACE: {compress_espace}, calibration iter {calib_iter}, variance {var}, oversampling {p}, compression iter: {compression_iter} , ratio: {ratio} ")

        # out_dir = f"/home/infres/tran-24/INSTANT/LBP-WHT/run_mains_compression/{dataset_name}_var{var}_oversampling{p}"
        # output_file_U = f"{out_dir}/U_dict.json"
        # output_file_U_grad = f"{out_dir}/U_grad_dict.json"
        # os.makedirs(out_dir, exist_ok=True)

        

        for i, data_batch in enumerate(self.data_loader):
    
            # compress vs compress_espace vs compress_random
            if compress_random:
                # random case

                # if i % compression_iter ==0:
                if i ==0:

                    self.model.disable_compression()
                    register_hooks(self.model)

                    images = data_batch['img']
                    labels = data_batch['gt_label']

                    # Pre-split batch into 2 parts (much faster than slicing in loop)
                    image_splits = torch.split(images, 32, dim=0)
                    label_splits = torch.split(labels, 32, dim=0)

                    for img_split, label_split in zip(image_splits, label_splits):

                        # No prints, no gc.collect()
                        sample = {
                            'img': img_split,
                            'gt_label': label_split
                        }

                        # Backward + hook collection
                        self.model.zero_grad(set_to_none=True)
                        outputs = self.model.train_step(sample, None, **kwargs)
                        outputs['loss'].backward()

                        # No need for empty_cache here
                        del outputs

                    for layer_name, input_sum in layer_inputs.items():
                        avg = input_sum
                        U_dict[layer_name] = random_gaussian_projector(avg, ratio)

                    for layer_name, grad_sum in layer_gradients.items():
                        avg = grad_sum
                        U_grad_dict[layer_name] = random_gaussian_projector(avg, ratio)

                    self.model.update_compression(U_dict, U_grad_dict)
                    unregister_hooks()

                    self._iter += 1
                
                else:
                    self.model.enable_compression()
                    # self.model.activate = torch.tensor(True, dtype=torch.bool)
                    self.data_batch = data_batch
                    self._inner_iter = i
                    self.call_hook('before_train_iter')

                    # start_forw_event = torch.cuda.Event(enable_timing=True)
                    # end_forw_event = torch.cuda.Event(enable_timing=True)
                    # start_back_opt_event = torch.cuda.Event(enable_timing=True)
                    # end_back_opt_event = torch.cuda.Event(enable_timing=True)
                    
                    # start_forw_event.record()
                    
                    self.run_iter(self.data_batch, train_mode=True, **kwargs)
                    
                    # end_forw_event.record()
                    # start_back_opt_event.record()
                    
                    self.call_hook('after_train_iter')
                    
                    # end_back_opt_event.record()
                    torch.cuda.synchronize()  # Single sync for both measurements (minimal overhead)
                    
                    # forward_time = start_forw_event.elapsed_time(end_forw_event)  # ms → seconds
                    # back_opt_time = start_back_opt_event.elapsed_time(end_back_opt_event)

                    # self.total_forward += forward_time
                    # self.total_backward_optimizer += back_opt_time
                    # self.total_backward += getattr(self, 'last_backward_time', 0)

                    del self.data_batch
                    self._iter += 1

            elif compress_espace or compress:
                flags = [compress, compress_espace]
                if sum(flags) != 1:
                    raise ValueError("Exactly one of compress, compress_espace must be True.")

                start_calib_event = torch.cuda.Event(enable_timing=True)
                end_calib_event = torch.cuda.Event(enable_timing=True)

                start_calib_event.record()

                if i % compression_iter in calib_steps:

                    # Enable hooks only once
                    if i % compression_iter == 0:
                        self.model.disable_compression()
                        register_hooks(self.model)

                    images = data_batch['img']
                    labels = data_batch['gt_label']

                    # Pre-split batch into 2 parts (much faster than slicing in loop)
                    image_splits = torch.split(images, 32, dim=0)
                    label_splits = torch.split(labels, 32, dim=0)

                    for img_split, label_split in zip(image_splits, label_splits):

                        # No prints, no gc.collect()
                        sample = {
                            'img': img_split,
                            'gt_label': label_split
                        }

                        # Backward + hook collection
                        self.model.zero_grad(set_to_none=True)
                        outputs = self.model.train_step(sample, None, **kwargs)
                        outputs['loss'].backward()

                        # No need for empty_cache here
                        del outputs

                    # Run compression update only at the last calibration iteration
                    if i % compression_iter == calib_iter - 1:

                        for layer_name, input_sum in layer_inputs.items():
                            avg = input_sum / calib_iter
                            if compress:
                                U_dict[layer_name] = SVD_expected_value(avg, var, p)
                            # if compress_random:
                            #     U_dict[layer_name] = random_gaussian_projector(avg, ratio)
                            if compress_espace:
                                U_dict[layer_name] = SVD_projection(avg, ratio)
                        for layer_name, grad_sum in layer_gradients.items():
                            avg = grad_sum / calib_iter
                            if compress:
                                U_grad_dict[layer_name] = SVD_expected_value(avg, var, p)
                            # if compress_random:
                            #     U_grad_dict[layer_name] = random_gaussian_projector(avg, ratio)
                            if compress_espace:
                                U_grad_dict[layer_name] = SVD_projection(avg, ratio)

                        self.model.update_compression(U_dict, U_grad_dict)
                        # log_svd_shapes(self._iter + 1, U_dict, output_file=output_file_U)
                        # log_svd_shapes(self._iter + 1, U_grad_dict, output_file=output_file_U_grad)
                        unregister_hooks()

                    self._iter += 1

                    # end_calib_event.record()
                    # self.total_calibration += start_calib_event.elapsed_time(end_calib_event) 

                else: 
                    self.model.enable_compression()
                    # self.model.activate = torch.tensor(True, dtype=torch.bool)
                    self.data_batch = data_batch
                    self._inner_iter = i
                    self.call_hook('before_train_iter')

                    start_forw_event = torch.cuda.Event(enable_timing=True)
                    end_forw_event = torch.cuda.Event(enable_timing=True)
                    start_back_opt_event = torch.cuda.Event(enable_timing=True)
                    end_back_opt_event = torch.cuda.Event(enable_timing=True)
                    
                    start_forw_event.record()
                    
                    self.run_iter(self.data_batch, train_mode=True, **kwargs)
                    
                    end_forw_event.record()
                    start_back_opt_event.record()
                    
                    self.call_hook('after_train_iter')
                    
                    end_back_opt_event.record()
                    torch.cuda.synchronize()  # Single sync for both measurements (minimal overhead)
                    
                    # forward_time = start_forw_event.elapsed_time(end_forw_event)  # ms → seconds
                    # back_opt_time = start_back_opt_event.elapsed_time(end_back_opt_event)

                    # self.total_forward += forward_time
                    # self.total_backward_optimizer += back_opt_time
                    # self.total_backward += getattr(self, 'last_backward_time', 0)

                    del self.data_batch
                    self._iter += 1
            else:
                # NORMAL TRAINING
                self.data_batch = data_batch
                self._inner_iter = i
                self.call_hook('before_train_iter')

                start_forw_event = torch.cuda.Event(enable_timing=True)
                end_forw_event = torch.cuda.Event(enable_timing=True)
                start_back_opt_event = torch.cuda.Event(enable_timing=True)
                end_back_opt_event = torch.cuda.Event(enable_timing=True)
                
                start_forw_event.record()
                
                self.run_iter(self.data_batch, train_mode=True, **kwargs)
                
                end_forw_event.record()
                start_back_opt_event.record()
                
                self.call_hook('after_train_iter')
                
                end_back_opt_event.record()
                torch.cuda.synchronize()  # Single sync for both measurements (minimal overhead)
                
                # forward_time = start_forw_event.elapsed_time(end_forw_event)  # ms → seconds
                # back_opt_time = start_back_opt_event.elapsed_time(end_back_opt_event)

                # self.total_forward += forward_time
                # self.total_backward_optimizer += back_opt_time
                # self.total_backward += getattr(self, 'last_backward_time', 0)
                
                del self.data_batch
                self._iter += 1

        self.call_hook('after_train_epoch')
        
        self._epoch += 1 # epoch is already run, start from 1
        print("Self epoch: ", self._epoch)
        print("Max epoch: ", self.max_epochs)
        # if self._epoch == self.max_epochs:
        #     print("Time reported for forward train_step: ", self.total_forward / self.max_epochs / 1000)
        #     print("Time reported for calibration: ", self.total_calibration / self.max_epochs / 1000)
        #     print("Time reported for backward+optimizer: ", self.total_backward_optimizer / self.max_epochs / 1000)
        #     print("Time reported for backward: ", self.total_backward / self.max_epochs / 1000)
    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        

        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    print("Len data_loaders: ", len(data_loaders[i]))
                    epoch_runner(data_loaders[i], **kwargs)


        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        # save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)
