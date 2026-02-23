from transformers import TrainerCallback
from itertools import islice
import gc
import torch.nn as nn
import torch

from SVD import SVD_expected_value
from utils import log_svd_shapes

P_dict = {}
Q_dict = {}
layer_inputs = {}
layer_gradients = {}
hook_handles_input = []
hook_handles_grad = []
def save_input_hook(name, module, input, output):
    input = input[0].detach()
    if input.dim() == 3: 
        X_XT =  torch.matmul( input,input.transpose(1, 2))
        average_X_XT = torch.mean(X_XT, dim=0)
    else:
        raise ValueError(
            f"Unexpected tensor dimensions for layer {name}: "
            f"got {input[0].dim()}, expected 3"
        )
    if name not in layer_inputs:
        layer_inputs[name] = average_X_XT
    else:
        layer_inputs[name] += average_X_XT
    del X_XT, average_X_XT
    torch.cuda.empty_cache()

def save_grad_hook(name, module, grad_input, grad_output):
    grad_output = grad_output[0].detach()
    if grad_output.dim() == 3: 
        gradX_gradXT =  torch.matmul(grad_output,grad_output.transpose(1, 2))
        average_gradX_gradXT = torch.mean(gradX_gradXT, dim=0)
    else:
        raise ValueError(
            f"Unexpected tensor dimensions for layer {name}: "
            f"got {grad_output[0].dim()}, expected 3"
        )

    if name not in layer_gradients:
        layer_gradients[name] = average_gradX_gradXT
    else:
        layer_gradients[name] += average_gradX_gradXT
    del gradX_gradXT, average_gradX_gradXT
    torch.cuda.empty_cache()

def register_hooks(model, model_name):
    global hook_handles_input, hook_handles_grad, hook_handles_weight
    for name, module in model.named_modules():
        if any(param.requires_grad for param in module.parameters()):
            if isinstance(module, nn.Linear):
                if model_name == "bert-base-uncased":
                    if "classifier" in name or "bert.embeddings" in name or "bert.pooler" in name:
                        continue
                elif model_name == "distilbert-base-uncased":
                    if "classifier" in name or "distilbert.embeddings" in name or "pre_classifier" in name:
                        continue
                handle_forward = module.register_forward_hook(lambda module, input, output, name=name: save_input_hook(name, module, input, output))
                handle_backward = module.register_backward_hook(lambda module, grad_input, grad_output, name=name: save_grad_hook(name, module, grad_input, grad_output))
                
                hook_handles_input.append(handle_forward)
                hook_handles_grad.append(handle_backward)

def unregister_hooks():
    global hook_handles_input, hook_handles_grad

    for handle in hook_handles_input:
        handle.remove()  # Unregister the input hook
    
    for handle in hook_handles_grad:
        handle.remove()  # Unregister the gradient hook

    hook_handles_input = []  # Clear the list of input hook handles
    hook_handles_grad = []  # Clear the list of gradient hook handles

    # Clear the layer input_sum and gradients dictionaries
    layer_inputs.clear()
    layer_gradients.clear()
    P_dict.clear()
    Q_dict.clear()
    gc.collect()
    torch.cuda.empty_cache()
    
class CustomLoggingCallback(TrainerCallback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_iter = None
    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataloader = kwargs.get('train_dataloader')
        if train_dataloader is not None:
            self.train_iter = iter(train_dataloader)
            
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.args.calib_iter == 0:
            if self.args.compress:
                print(f"[CALIBRATION] Step {state.global_step}, Epoch {state.epoch}")
                model = kwargs.get('model')
                if model is None:
                    print("In Calibration CallBack: Model not found in kwargs!")
                    return
                loss_fn = nn.CrossEntropyLoss()
                optimizer = kwargs.get('optimizer')

                if optimizer is None:
                    print("Optimizer not found in kwargs!")
                model.disable_compression()
                register_hooks(model, self.args.model_name)

                calib_iter = self.args.calib_batches
                calib_batches = []

                for batch in islice(self.train_iter, calib_iter):
                    calib_batches.append(batch)

                for calib_batch in calib_batches:
                    num_splits = self.args.num_split
                    inputs = {k: v.to(model.device) for k, v in calib_batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
                    splits = {
                        k: torch.chunk(v, num_splits, dim=0)
                        for k, v in inputs.items()
                    }
                    for i in range(num_splits):
                        optimizer.zero_grad()
                        sub_inputs = { k: splits[k][i] for k in splits }
                        outputs = model(**sub_inputs)
                        logits = outputs.logits
                        labels = sub_inputs['labels']
                        loss = loss_fn(logits, labels)
                        loss.backward()

                    del outputs, logits, loss, labels
                    gc.collect()
                    torch.cuda.empty_cache()
                
                del calib_batches
                gc.collect()
                torch.cuda.empty_cache()

                for layer_name, input_sum in layer_inputs.items():  
                    if self.args.model_name == "bert-base-uncased":
                        if layer_name.startswith('bert.encoder.layer.'):
                            average_X_XT = input_sum / calib_iter/num_splits

                            P_dict[layer_name] = SVD_expected_value(average_X_XT, self.args.var, self.args.over_sampling)
                    if self.args.model_name == "distilbert-base-uncased":
                        if layer_name.startswith('distilbert.transformer.layer.'):
                            average_X_XT = input_sum / calib_iter/num_splits

                            P_dict[layer_name] = SVD_expected_value(average_X_XT, self.args.var, self.args.over_sampling)

                for layer_name, grad_sum in layer_gradients.items():         
                    if self.args.model_name == "bert-base-uncased":
                        if layer_name.startswith('bert.encoder.layer.'):
                            average_grad_X_XT = grad_sum / calib_iter
                            Q_dict[layer_name] = SVD_expected_value(average_grad_X_XT, self.args.var, self.args.over_sampling)
                    if self.args.model_name == "distilbert-base-uncased":
                        if layer_name.startswith('distilbert.transformer.layer.'):
                            average_grad_X_XT = grad_sum / calib_iter
                            Q_dict[layer_name] = SVD_expected_value(average_grad_X_XT, self.args.var, self.args.over_sampling)
                output_file_U = f"{self.args.output_dir}/P_dict.json"
                output_file_U_grad = f"{self.args.output_dir}/Q_dict.json"
                log_svd_shapes(state.global_step, P_dict, output_file=output_file_U)
                log_svd_shapes(state.global_step, Q_dict, output_file=output_file_U_grad)
                
                model.update_compression(P_dict, Q_dict)
                model.cuda()
                model.enable_compression()
                del average_X_XT, average_grad_X_XT
                unregister_hooks()
                gc.collect()
                torch.cuda.empty_cache()
                calib_steps = list(range(self.args.calib_batches))
                if state.global_step % self.args.calib_iter not in calib_steps:
                    next(self.train_iter)