# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import random
import warnings
import json

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (
    auto_select_device,
    collect_env,
    get_root_logger,
    setup_multi_processes,
)

# from custom_op.register import register_filter
from functools import reduce
from custom_op.linear_lora import LinearLORA
from custom_op.pointwise_conv_lora import PointwiseConvLORA
import torch.nn as nn
from utils.logging import get_active_branch_name, get_latest_log_version
from custom_op.INSTANT import register_filter, register_filter_espace

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--load-from", help="the checkpoint file to load weights from")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--device", help="device used for training. (Deprecated)")
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="(Deprecated, please use --gpu-id) number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--ipu-replicas", type=int, default=None, help="num of ipu replicas to use"
    )
    parser.add_argument("--seed", type=int, default=233, help="random seed")
    parser.add_argument(
        "--diff-seed",
        action="store_true",
        help="Whether or not set different seeds for different ranks",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--log-postfix", type=str, default="", help="postfix of the log dir"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress training or not"
    )
    
    parser.add_argument(
        "--compress_random",
        action="store_true",
        help="Compress with Gaussian matrix with ratio, not INSTANT"
    )

    parser.add_argument(
        "--compress_espace",
        action="store_true",
        help="Compress with INSTANT scheme but choose fixed bases based on ratio"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs-main",
        help="output dir"
    )

    parser.add_argument(
        "--var",
        type=float,
        default=0.95,
        help="explainable variance"
    )

    parser.add_argument(
        "--calib_iter",
        type=int,
        default=5,
        help="iterations used to calibrate"
    )

    parser.add_argument(
        "--ratio",
        type=int,
        default=2,
        help="ratio used for random_instant and instant_espace"
    )

    parser.add_argument(
        "--over_sam",
        type=int,
        default=0,
        help="Over sampling to compression tensors"
    )

    parser.add_argument(
        "--compression_iter",
        type=int,
        default=200,
        help="Number of iterations using compression"
    )

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

def add_activate_to_module():
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


# Add P, Q to the model
def add_compression_tensor_to_module():
    # Define a method to register buffers for a module instance
    def register_compression_buffers(module):
        # Register buffers with None as placeholder
        if not hasattr(module, 'CompressionTensor_x'):
            module.register_buffer('CompressionTensor_x', None)
        if not hasattr(module, 'CompressionTensor_gy'):
            module.register_buffer('CompressionTensor_gy', None)
        # if not hasattr(module, 'energy_preserve_P'):
        #     module.register_buffer('energy_preserve_P', None)
        # if not hasattr(module, 'energy_preserve_Q'):
        #     module.register_buffer('energy_preserve_Q', None)

    # Override __init__ of nn.Module to register buffers for all new modules
    original_init = nn.Module.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        register_compression_buffers(self)
    nn.Module.__init__ = new_init

    # Add update_compression method to nn.Module
    def update_compression(self, P_dict, Q_dict):
        for name, module in self.named_modules():
            if name in P_dict and name in Q_dict:

                compression_x = P_dict[name].detach().requires_grad_(False)
                compression_gy = Q_dict[name].detach().requires_grad_(False)


                # Validate that tensors are 2D
                assert compression_x.dim() == 2, f"CompressionTensor_x for {name} is not 2D"
                assert compression_gy.dim() == 2, f"CompressionTensor_gy for {name} is not 2D"

                # Register updated buffers
                module.register_buffer('CompressionTensor_x', compression_x)
                module.register_buffer('CompressionTensor_gy', compression_gy)
        return self
    nn.Module.update_compression = update_compression

def get_svd_layers(model):
    modules_compressed=[]
    for name, module in model.named_modules():
        if any(param.requires_grad for param in module.parameters()):
            if isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)):
                # Skip layers with "classifier" and "embeddings" in their name
                if "classifier" in name or "head." in name or "cpb_mlp" in name or "downsample" in name :
                    continue
                modules_compressed.append(name)
    return modules_compressed

def get_moment_logger(model, name):
    model.moment1[name] = 0.0
    model.moment2[name] = 0.0
    model.moment_step[name] = 0

    def _logger(grad):
        model.moment1[name] += (grad - model.moment1[name]) / (
            model.moment_step[name] + 1
        )
        model.moment2[name] += (grad.square() - model.moment2[name]) / (
            model.moment_step[name] + 1
        )
        model.moment_step[name] += 1

    return _logger


def main():
    time.sleep(random.random())
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            "`--gpus` is deprecated because we only support "
            "single GPU mode in non-distributed training. "
            "Use `gpus=1` now."
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed training. Use the first GPU "
            "in `gpu_ids` now."
        )
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.ipu_replicas is not None:
        cfg.ipu_replicas = args.ipu_replicas
        args.device = "ipu"

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
        try:
            cfg.evaluation.pop("gpu_collect")
        except Exception:
            pass
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    if cfg.get("gradient_filter", None) is None:
        cfg.gradient_filter = dict(enable=False, filter_install=[])
    if cfg.get("freeze_layers", None) is None:
        cfg.freeze_layers = []

    sync_objects = [None, None]
    if not distributed or dist.get_rank() == 0:
        # work_dir is determined in this priority: CLI > segment in file > filename
        work_dir = "./runs"
        branch_name = get_active_branch_name()
        if branch_name:
            work_dir += "-" + branch_name
        postfix = f"_{args.log_postfix}" if args.log_postfix != "" else ""
        log_name = osp.splitext(osp.basename(args.config))[0] + postfix
        if args.work_dir is not None:
            work_dir = args.work_dir
        work_dir = osp.join(work_dir, log_name)
        max_version = get_latest_log_version(work_dir)
        work_dir = osp.join(work_dir, f"version_{max_version}")
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        cfg.work_dir = work_dir

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        # init the logger before other steps
        log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
        sync_objects = [timestamp, work_dir]
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        if distributed:
            dist.broadcast_object_list(sync_objects, src=0)
    else:
        dist.broadcast_object_list(sync_objects, src=0)
        timestamp, work_dir = sync_objects
        cfg.work_dir = work_dir
        log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    cfg.device = args.device or auto_select_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed

    # # set config
    cfg.compress = args.compress
    cfg.var = args.var
    cfg.over_sam = args.over_sam
    cfg.calib_iter = args.calib_iter
    cfg.compression_iter = args.compression_iter
    cfg.compress_random = args.compress_random
    cfg.compress_espace = args.compress_espace
    cfg.ratio = args.ratio

    model = build_classifier(cfg.model)
    model.init_weights()


    for layer_path in cfg.freeze_layers:
        active = layer_path[0] == "~"
        if active:
            layer_path = layer_path[1:]
            logger.info(f"Unfreeze: {layer_path}")
        else:
            logger.info(f"Freeze: {layer_path}")
        path_seq = layer_path.split(".")

        target = reduce(getattr, path_seq, model)
        if isinstance(target, nn.Parameter):
            target.requires_grad = active
        else:
            for p in target.parameters():
                p.requires_grad = active

    logger.info("Model:")
    logger.info(str(model))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # save mmcls version, config file content and class names in
    # runner as meta data
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )
    )
    if args.compress or args.compress_random:
        add_activate_to_module()
        add_compression_tensor_to_module()

        compress_list = get_svd_layers(model)
        model = register_filter(model, compress_list)

    if args.compress_espace:
        add_activate_to_module()
        add_compression_tensor_to_module()

        compress_list = get_svd_layers(model)
        model = register_filter_espace(model, compress_list)

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device=cfg.device,
        meta=meta,
    )


if __name__ == "__main__":
    main()
