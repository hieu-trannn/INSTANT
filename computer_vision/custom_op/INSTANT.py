import torch as th
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from typing import Any
import math

class LinearSVDOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        # !!! How to deal with bias?
        x, weight, bias, CompressionTensor_x, CompressionTensor_gy  = args
        
        if x.shape[2] < x.shape[1]:
            # compressionTensor_x: (r, Ci)
            x_compressed = th.einsum("blc,rc->blr", x, CompressionTensor_x)
        else:
        # x_compressed of calibrating with x, CompressionTensor_x: (r, L)
            x_compressed = CompressionTensor_x@x

        y = F.linear(x, weight, bias)
        # cfgs = th.tensor([bias is not None])

        """Save for backward"""
        # if ctx is not None:
        ctx.input_shape = x.shape
        ctx.save_for_backward(x_compressed, weight, th.tensor([bias is not None]), CompressionTensor_x, CompressionTensor_gy )
        # ctx.save_for_backward(x, weight, th.tensor([bias is not None]), CompressionTensor_x, CompressionTensor_gy )
        
        # ctx.save_for_backward(x, weight, cfgs, CompressionTensor_x, CompressionTensor_gy )
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # get info from forward
        shape_x = ctx.input_shape 
        x_compressed, weight, has_bias, CompressionTensor_x, CompressionTensor_gy  = ctx.saved_tensors
        # x, weight, has_bias, CompressionTensor_x, CompressionTensor_gy = ctx.saved_tensors
        # x, weight, has_bias, CompressionTensor_x, CompressionTensor_gy = ctx.saved_tensors

        # (batch_size, output_channels, height, width)
        (grad_y,) = grad_outputs
        # """Calibrate with X and W"""
        # g_w=(g_y^T.(X.P^T )).P  
        # grad_w = ((grad_y.transpose(1,2) @ x_compressed) @ CompressionTensor_x ).sum(dim=0)
        if grad_y.shape[2] < grad_y.shape[1]:
            # grad_x = (grad_y @ CompressionTensor_gy.T)@ (CompressionTensor_gy@ weight) * (1/energy_preserve_Q)
            grad_y_compress = th.einsum("blo,ro->blr", grad_y, CompressionTensor_gy)
            weight_compress = th.einsum("ro,oi->ri", CompressionTensor_gy, weight)
            grad_x = th.einsum("blr,ri->bli", grad_y_compress, weight_compress)
        else:
            # grad_x = CompressionTensor_gy.T @ ((CompressionTensor_gy @ grad_y)@ weight) * (1/energy_preserve_Q)
            grad_y_compress = th.einsum("rl, blo ->bro", CompressionTensor_gy, grad_y)
            weight_compress = th.einsum("bro,oi->bri", grad_y_compress, weight)
            grad_x = th.einsum("rl,bri->bli", CompressionTensor_gy, weight_compress)

        if shape_x[2] < shape_x[1]:
            grad_w_temp_1 = th.einsum('blo,blr->bor', grad_y, x_compressed)
            grad_w = th.einsum('bor,ri->oi', grad_w_temp_1, CompressionTensor_x)
            # grad_w = ((grad_y.transpose(1,2) @ x_compressed) @CompressionTensor_x).sum(dim=0)* (1/energy_preserve_P)
        else:
            grad_w_temp_1 = th.einsum('blo,rl->bor', grad_y, CompressionTensor_x)
            grad_w = th.einsum('bor,bri->oi', grad_w_temp_1, x_compressed)
            # grad_w = ((grad_y.transpose(1,2) @ CompressionTensor_x.T) @ x_compressed).sum(dim=0) * (1/energy_preserve_P)
        # grad_w = (grad_y.transpose(1,2) @ x).sum(dim=0)

        # !!! How to deal with bias
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 1))
        else:
            grad_b = None
        # grad_x, grad_w: in full rank space
        return grad_x, grad_w, grad_b, None, None, None, None, None
    

class LinearCompressClass(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # self.activate = activate
        
    def forward(self, input: th.Tensor) -> th.Tensor:
        # if self.activating and self.training and (self.CompressionTensor_x is not None):
        if self.activating and self.training:
            y = LinearSVDOp.apply(
                input,
                self.weight,
                self.bias,
                self.CompressionTensor_x.T.clone().detach().requires_grad_(False).cuda(),
                self.CompressionTensor_gy.T.clone().detach().requires_grad_(False).cuda(),
            )
        else:
            # print("Activate and training:", self.activating, self.training)
            y = super().forward(input)

        return y

class PointwiseConvSVDOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        # !!! How to deal with bias?
        input, weight, bias, CompressionTensor_x, CompressionTensor_gy = args
        b, _, h, w = input.shape
        c_o, c_i = weight.shape[:2]
        # reshape x, w
        weight = th.squeeze(weight)
        x = input.view(b,c_i,h*w).transpose(1,2) # shape: (b, L, C_i)
        # get output by low-rank multiplication

        # """Calibrate with X"""
        # x_compressed shape: (B, R1, Ci)

        # print("CompressionTensor_x: ", CompressionTensor_x.shape)
        # print("x: ", x.shape)
        # print("weight: ", weight.shape)
        if x.shape[2] < x.shape[1]:
            x_compressed = th.einsum("blc,rc->blr", x, CompressionTensor_x)
        else:
            x_compressed = CompressionTensor_x@x
        y = F.linear(x, weight, bias)
        y = y.transpose(1,2).view(b, c_o, h, w)
        cfgs = th.tensor([bias is not None,h,w])

        """Save for backward"""
        # if ctx is not None:
        ctx.input_shape = x.shape
        ctx.save_for_backward(x_compressed, weight, cfgs, CompressionTensor_x, CompressionTensor_gy )

        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # get info from forward
        shape_x = ctx.input_shape 
        x_compressed, weight, cfgs, CompressionTensor_x, CompressionTensor_gy  = ctx.saved_tensors
        # x, weight, cfgs, CompressionTensor_x, CompressionTensor_gy  = ctx.saved_tensors
        # x, weight, cfgs, CompressionTensor_x, CompressionTensor_gy = ctx.saved_tensors
        # (batch_size, output_channels, height, width)
        (grad_y,) = grad_outputs
        # b, R1, c_i  = x_compressed.shape
        b, L, c_i = shape_x
        c_o, c_i = weight.shape
        has_bias,h,w = cfgs

        # reshape gy to linear
        grad_y = grad_y.view(b,c_o,-1).transpose(1,2)

        # grad_x original
        # grad_x = grad_y@weight

        # gx =  Q^T.((Q.g_y ).W)
        if grad_y.shape[2] < grad_y.shape[1]:
            # grad_x = (grad_y @ CompressionTensor_gy.T)@ (CompressionTensor_gy@ weight) * (1/energy_preserve_Q)
            grad_y_compress = th.einsum("blo,ro->blr", grad_y, CompressionTensor_gy)
            weight_compress = th.einsum("ro,oi->ri", CompressionTensor_gy, weight)
            grad_x = th.einsum("blr,ri->bli", grad_y_compress, weight_compress)
        else:
            # grad_x = CompressionTensor_gy.T @ ((CompressionTensor_gy @ grad_y)@ weight) * (1/energy_preserve_Q)
            grad_y_compress = th.einsum("rl, blo ->bro", CompressionTensor_gy, grad_y)
            weight_compress = th.einsum("bro,oi->bri", grad_y_compress, weight)
            grad_x = th.einsum("rl,bri->bli", CompressionTensor_gy, weight_compress)
        
        if shape_x[2] < shape_x[1]:
            grad_w_temp_1 = th.einsum('blo,blr->bor', grad_y, x_compressed)
            grad_w = th.einsum('bor,ri->oi', grad_w_temp_1, CompressionTensor_x)
            # grad_w = ((grad_y.transpose(1,2) @ x_compressed) @CompressionTensor_x).sum(dim=0)* (1/energy_preserve_P)
        else:
            grad_w_temp_1 = th.einsum('blo,rl->bor', grad_y, CompressionTensor_x)
            grad_w = th.einsum('bor,bri->oi', grad_w_temp_1, x_compressed)


        # reshape grad_x, grad_w
        grad_x = grad_x.transpose(1, 2).view(b,c_i, h, w)
        grad_w = grad_w.view(c_o, c_i, 1, 1)

        # !!! How to deal with bias
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 1))
        else:
            grad_b = None
        # grad_x, grad_w: in full rank space
        return grad_x, grad_w, grad_b, None, None, None, None


class PointwiseConvCompressClass(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activate = False):
        """
        Initialize the PointwiseConvCompressClass.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolution kernel (default: 1 for pointwise)
            stride (int or tuple): Stride of the convolution
            padding (int or tuple): Padding added to all sides of the input
            dilation (int or tuple): Spacing between kernel elements
            groups (int): Number of blocked connections from input to output channels
            bias (bool): If True, adds a learnable bias to the output
            padding_mode (str): Padding mode ('zeros', 'reflect', etc.)
        """
        # Call the parent class (nn.Conv2d) constructor
        super(PointwiseConvCompressClass, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    
    def forward(self, input: th.Tensor) -> th.Tensor:
        # !!!! NEED THIS?
        # if self.activating and self.training and (self.CompressionTensor_x is not None):
        if self.activating and self.training:
            y = PointwiseConvSVDOp.apply(
                input,
                self.weight,
                self.bias,
                self.CompressionTensor_x.T.clone().detach().requires_grad_(False).cuda(),
                self.CompressionTensor_gy.T.clone().detach().requires_grad_(False).cuda(),
            )
        # elif self.training:
            # y = super().forward(input)
        else:
            # y = PointwiseConvSVDOp.forward(None, input, self.weight, self.bias, self.CompressionTensor_x, self.CompressionTensor_gy)
            y = super().forward(input)
        return y
    

class LinearCompressClassEspace(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, input: th.Tensor) -> th.Tensor:
        if self.activating and self.training:
            CompressionTensor_x = self.CompressionTensor_x.T.clone().detach().requires_grad_(False).cuda()

            if input.shape[2] < input.shape[1]:
                x_compressed = th.einsum("blc,rc->blr", input, CompressionTensor_x)
                w_compressed = CompressionTensor_x @ self.weight.T
                y = x_compressed @ w_compressed + self.bias
            else:
            # x_compressed of calibrating with x, CompressionTensor_x: (r, L)
                x_compressed = CompressionTensor_x@input
                y = (CompressionTensor_x.T @ x_compressed) @ self.weight.T + self.bias

        else:
            y = super().forward(input)

        return y

class PointwiseConvCompressClassEspace(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', activate = False):
        """
        Initialize the PointwiseConvCompressClass.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolution kernel (default: 1 for pointwise)
            stride (int or tuple): Stride of the convolution
            padding (int or tuple): Padding added to all sides of the input
            dilation (int or tuple): Spacing between kernel elements
            groups (int): Number of blocked connections from input to output channels
            bias (bool): If True, adds a learnable bias to the output
            padding_mode (str): Padding mode ('zeros', 'reflect', etc.)
        """
        # Call the parent class (nn.Conv2d) constructor
        super(PointwiseConvCompressClassEspace, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, input: th.Tensor) -> th.Tensor:
        # !!!! NEED THIS?
        # if self.activating and self.training and (self.CompressionTensor_x is not None):
        if self.activating and self.training:

            b, c_i,h,w = input.shape
            input = input.view(b,c_i,h*w).transpose(1,2) # shape: (b, L, C_i)

            weight = th.squeeze(self.weight)

            c_o = weight.shape[0]

            CompressionTensor_x = self.CompressionTensor_x.T.clone().detach().requires_grad_(False).cuda()

            if input.shape[2] < input.shape[1]:
                x_compressed = th.einsum("blc,rc->blr", input, CompressionTensor_x)
                w_compressed = CompressionTensor_x @ weight.T
                y = x_compressed @ w_compressed + self.bias
                y = y.transpose(1,2).view(b, c_o, h, w)
            else:
            # x_compressed of calibrating with x, CompressionTensor_x: (r, L)
                x_compressed = CompressionTensor_x@input
                y = (CompressionTensor_x.T @ x_compressed) @ weight.T + self.bias
                y = y.transpose(1,2).view(b, c_o, h, w)
        else:
            y = super().forward(input)
        return y

def wrap_pointwise_conv_compression_layer(
    conv_layer: nn.Conv2d, **kwargs
):
    new_op = PointwiseConvCompressClass(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
    )
    new_op.weight.data = conv_layer.weight.data
    if new_op.bias is not None:
        new_op.bias.data = conv_layer.bias.data
    return new_op

def wrap_linear_compression_layer(
    linear_layer: nn.Linear, **kwargs
):
    new_linear = LinearCompressClass(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
    )
    new_linear.weight.data = linear_layer.weight.data
    if linear_layer.bias is not None:
        new_linear.bias.data = linear_layer.bias.data
    return new_linear


# Register matrix filter for specific model name: backbone...
def register_filter(model, modules_compressed):
    for name, module in model.named_modules():
        if name in modules_compressed:
            if module.__class__.__name__ == "Linear":
                compressed_module = wrap_linear_compression_layer(module)
            elif module.__class__.__name__ == "Conv2d" and module.kernel_size == (1, 1):
                compressed_module = wrap_pointwise_conv_compression_layer(module)
            else:
                print("View compressed module in Register Filter again!!")
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent_module, name.rsplit('.', 1)[1], compressed_module)  
    
    return model


def wrap_pointwise_conv_compression_layer_espace(
    conv_layer: nn.Conv2d, **kwargs
):
    new_op = PointwiseConvCompressClassEspace(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
    )
    new_op.weight.data = conv_layer.weight.data
    if new_op.bias is not None:
        new_op.bias.data = conv_layer.bias.data
    return new_op

def wrap_linear_compression_layer_espace(
    linear_layer: nn.Linear, **kwargs
):
    new_linear = LinearCompressClassEspace(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
    )
    new_linear.weight.data = linear_layer.weight.data
    if linear_layer.bias is not None:
        new_linear.bias.data = linear_layer.bias.data
    return new_linear

# Register matrix filter for specific model name: backbone...
def register_filter_espace(model, modules_compressed):
    for name, module in model.named_modules():
        if name in modules_compressed:
            if module.__class__.__name__ == "Linear":
                compressed_module = wrap_linear_compression_layer_espace(module)
            elif module.__class__.__name__ == "Conv2d" and module.kernel_size == (1, 1):
                compressed_module = wrap_pointwise_conv_compression_layer_espace(module)
            else:
                print("View compressed module in Register Filter again!!")
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] 
            setattr(parent_module, name.rsplit('.', 1)[1], compressed_module)  
    
    return model
    # pass
