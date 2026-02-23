import torch
import torch.nn as nn

def add_activate_to_module(): 
    if not hasattr(nn.Module, 'activate'):
        nn.Module.activate = True 

    def enable_compression(self):
        self.activate = True
        for module in self.children():
            module.enable_compression()
        return self
    nn.Module.enable_compression = enable_compression

    def disable_compression(self):
        self.activate = False
        for module in self.children():
            module.disable_compression()
        return self
    nn.Module.disable_compression = disable_compression

def add_compression_tensor_to_module():
    if not hasattr(nn.Module, 'CompressionTensor_x'):
        nn.Module.CompressionTensor_x = None 
    if not hasattr(nn.Module, 'CompressionTensor_gy'):
        nn.Module.CompressionTensor_gy = None 
        
    def update_compression(self, P_dict, Q_dict):
        for name, module in self.named_modules():
            if name in P_dict and name in Q_dict:
                module.CompressionTensor_x = P_dict[name].detach().requires_grad_(False)
                module.CompressionTensor_gy = Q_dict[name].detach().requires_grad_(False)
        return self
    nn.Module.update_compression = update_compression
    
def freeze_bert_layers(model, n_last_layers, model_name):
    """
    Freezes all BERT layers except the last n transformer layers and the classifier.
    
    Args:
        model: The BertForClassification model.
        n_last_layers (int): Number of last transformer layers to keep trainable.
        model_name: Name of the model
    """
    if n_last_layers != 0:
        if model_name == "bert-base-uncased":
            n_last_layers = min(n_last_layers, 12)
            trainable_layer_indices = list(range(12 - n_last_layers, 12))
            
            for name, param in model.named_parameters():
                is_trainable = (
                    any(f"bert.encoder.layer.{i}" in name for i in trainable_layer_indices) or
                    name.startswith("classifier")
                )
                param.requires_grad = is_trainable
        elif model_name == "distilbert-base-uncased":
            n_last_layers = min(n_last_layers, 6)
            trainable_layer_indices = list(range(6 - n_last_layers, 6))
            
            for name, param in model.named_parameters():
                is_trainable = (
                    any(f"distilbert.transformer.layer.{i}" in name for i in trainable_layer_indices) or
                    name.startswith("classifier")
                )
                param.requires_grad = is_trainable
                
def get_svd_layers(model): #Get the name of the layer that applied compression on
    modules_compressed=[]
    for name, module in model.named_modules():
        if any(param.requires_grad for param in module.parameters()):
            if isinstance(module, nn.Linear):
                if "classifier" in name or "bert.embeddings" in name or "bert.pooler" in name:
                    continue
                modules_compressed.append(name)
    return modules_compressed

def print_status(model):
    for name, param in model.named_parameters():
        status = "❄️ Frozen" if not param.requires_grad else "🔥 Unfrozen"
        print(f"{name}: {status}")