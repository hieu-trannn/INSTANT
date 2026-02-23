import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import linear, conv2d, avg_pool2d, pad
import torch.nn as nn
from math import ceil, sqrt
import time

class Conv2dAvgOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, order, shape  = args
        assert x.dim() == 3, f"{x.shape}"
        b = x.shape[0]
        # print("shape inside: ", shape)
        h, w = shape
        c_o, c_i = weight.shape
        
        y = linear(x, weight, bias)
        # print("y: ", y.shape)

        start = time.time()
        p_h, p_w = ceil(h / order), ceil(w / order)
        x = x.view(-1, h, w, c_i).permute(0, 3, 1, 2)
        # print("x: ", x.shape)
        # x, weight, bias, stride, dilation, padding, order, groups = args
        x_h, x_w = x.shape[-2:]
        # y = conv2d(x, weight, bias, stride, padding,
                #    dilation=dilation, groups=groups)
  
        p_h, p_w = ceil(h / order), ceil(w / order)
        
        # weight_sum = weight.sum(dim=(-1, -2))
        x_order_h, x_order_w = order, order
        x_pad_h, x_pad_w = ceil(
            (p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

        x_sum = avg_pool2d(x, kernel_size=(x_order_h, x_order_w),
                           stride=(x_order_h, x_order_w),
                           padding=(x_pad_h, x_pad_w), divisor_override=1)
        # print("x sum: ", x_sum.shape)
        end = time.time()
        # print(f"Compression time of GF in the forward: {end-start}")
        cfgs = th.tensor([bias is not None,
                          x_pad_h, x_pad_w,
                          x_h, x_w, order])
        ctx.save_for_backward(x_sum, weight, cfgs)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight_sum, cfgs = ctx.saved_tensors
        has_bias,\
            x_pad_h, x_pad_w,\
            x_h, x_w, order = [int(c) for c in cfgs]
        n, c_in, p_h, p_w = x_sum.shape
        c_o, c_i = weight_sum.shape
        grad_y, = grad_outputs
        grad_y_2d = grad_y.view(-1, x_h, x_w, c_o).permute(0, 3, 1, 2)
        # print("grad y 2d: ", grad_y_2d.shape)
        _, c_out, gy_h, gy_w = grad_y_2d.shape

        start = time.time()

        grad_y_pad_h, grad_y_pad_w = ceil(
            (p_h * order - gy_h) / 2), ceil((p_w * order - gy_w) / 2)
        grad_y_avg = avg_pool2d(grad_y_2d, kernel_size=order, stride=order,
                                padding=(grad_y_pad_h, grad_y_pad_w),
                                count_include_pad=False)
        # print("grad_y_avg: ", grad_y_avg.shape)
        grad_x_sum = (
            weight_sum.t() @ grad_y_avg.flatten(start_dim=2)).view(n, c_in, p_h, p_w)
        # print("grad x sum: ", grad_x_sum.shape)
        gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
        # print("gy: ", gy.shape)
        gx = x_sum.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
        # print("gx: ", gx.shape)
        grad_w = gy @ gx
        # print("grad w: ", grad_w.shape)
        # grad_w = th.broadcast_to(grad_w_sum.view(
        #     c_out, c_in, 1, 1), (c_out, c_in, 1, 1)).clone()

        grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1),
                                 (n, c_in, p_h, p_w, order * 1, order * 1))
        grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
            n, c_in, p_h * order * 1, p_w * order * 1)
        grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]
        grad_x = grad_x.flatten(start_dim=-2).permute(0, 2, 1)
        end = time.time()
        # print("Calculation time of GF's backward: ", end - start)
        if has_bias:
            grad_b = grad_y_2d.sum(dim=(0, 2, 3))
        else:
            grad_b = None

        return grad_x, grad_w, grad_b, None, None


class Conv2dDilatedOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, stride, dilation, padding, order, groups = args
        x_h, x_w = x.shape[-2:]
        k_h, k_w = weight.shape[-2:]
        y = conv2d(x, weight, bias, stride, padding,
                   dilation=dilation, groups=groups)
        h, w = y.shape[-2:]
        p_h, p_w = ceil(h / order), ceil(w / order)
        x_order_h, x_order_w = order * stride[0], order * stride[1]
        x_pad_h, x_pad_w = ceil(
            (p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)
        x_sum = avg_pool2d(x, kernel_size=(x_order_h, x_order_w),
                           stride=(x_order_h, x_order_w),
                           padding=(x_pad_h, x_pad_w), divisor_override=1)
        cfgs = th.tensor([bias is not None, groups != 1,
                          stride[0], stride[1],
                          x_pad_h, x_pad_w,
                          k_h, k_w,
                          x_h, x_w, order, dilation[0]])
        ctx.save_for_backward(x_sum, weight, cfgs)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight, cfgs = ctx.saved_tensors
        has_bias, grouping,\
            s_h, s_w,\
            x_pad_h, x_pad_w,\
            k_h, k_w,\
            x_h, x_w, order, dil = [int(c) for c in cfgs]
        n, c_in, p_h, p_w = x_sum.shape
        grad_y, = grad_outputs
        _, c_out, gy_h, gy_w = grad_y.shape
        grad_y_pad_h, grad_y_pad_w = ceil(
            (p_h * order - gy_h) / 2), ceil((p_w * order - gy_w) / 2)
        grad_y_avg = avg_pool2d(grad_y, kernel_size=order, stride=order,
                                padding=(grad_y_pad_h, grad_y_pad_w),
                                count_include_pad=False)
        equ_dil = dil // order
        if grouping:
            rot_weight = th.flip(weight, (2, 3))
            grad_x_sum = conv2d(grad_y_avg, rot_weight, padding=equ_dil,
                                dilation=equ_dil, groups=weight.shape[0])
            grad_w_sum = (x_sum * grad_y_avg).sum(dim=(0, 2, 3))
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
        else:
            rot_weight = th.flip(weight.permute(1, 0, 2, 3), (2, 3))
            grad_x_sum = conv2d(grad_y_avg, rot_weight,
                                padding=equ_dil, dilation=equ_dil)
            gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
            gx = x_sum.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
            grad_w_sum = gy @ gx
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1),
                                 (n, c_in, p_h, p_w, order * s_h, order * s_w))
        grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
            n, c_in, p_h * order * s_h, p_w * order * s_w)
        grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]
        # print("grad x: ", grad_x.shape)
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 2, 3))
        else:
            grad_b = None

        return grad_x, grad_w, grad_b, None, None, None, None, None


class LinAvg(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            order=4,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            shape=None
    ) -> None:

        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(LinAvg, self).__init__(in_features=in_features,
                                        out_features=out_features,
                                        bias=bias,
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.order = order
        self.shape = shape

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.shape is None:
            # print("x shape: ", x.shape)
            if x.dim() == 4:
                self.shape = x.shape[1:3]
            elif x.dim() == 3:
                s = int(sqrt(x.shape[1]))
                self.shape = [s, s]
                # print("shape: ", self.shape)
        flatten = x.dim() == 4
        if flatten:
            x = x.flatten(start_dim=1, end_dim=3)

        if self.activate:
            y = Conv2dAvgOp.apply(x, self.weight, self.bias, self.order, self.shape)
        else:
            y = super().forward(x)
        return y



def wrap_linear_layer(conv, radius, active):
    new_conv = LinAvg(in_features=conv.in_features,
                         out_features=conv.out_features,
                         bias=conv.bias is not None,
                         order=radius,
                         activate=active
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv