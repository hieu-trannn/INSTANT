# Computer Vision: Setup & Calibration

This directory contains the computer vision experiments. Our codebase builds upon the [LBP-WHT framework](https://github.com/SLDGroup/LBP-WHT/tree/main).

## 🛠️ 1. Environment Setup

We recommend using Conda to manage your environment. 

1. Create and activate conda environment
```bash
conda create -n instant python=3.8 -y
conda activate instant
```
2. Install Pytorch 1.13.1 with CUDA 11.6:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
3. Install mmcv-full and other dependencies
```bash
pip install openmim
mim install mmcv-full==1.7.0
cd mmclassification
pip install -e .
pip install yapf==0.32.0
pip install scipy future tensorboard
```

4. Dataset preparation

- Create a data folder under the root of the repo
- Cifar10/100 will be download and preprocessed automatically.
- For Flowers102, Food101, Pets, you need to manually download and place them under the data folder
    - The Flowers102 dataset need to be preprocessed with tools/convert_flowers_102.py

## 📦 2. Pretrained Models
Before running the evaluation or training scripts, you need to download the required pretrained models, put under `pretrained_ckpts` directory.

1. EfficientFormer L1:
```bash
mim download mmcls --config efficientformer-l1_3rdparty_8xb128_in1k --dest pretrained_ckpts
```
Need to convert the downloaded checkpoint with `tools/correct_efficientformer_ckpt.py`

2. EfficientFormerV2S0: Download manually [here](https://drive.google.com/file/d/1PXb7b9pv9ZB4cfkRkYEdwgWuVwvEiazq/view)

3. SwinV2
```bash
mim download mmcls --config swinv2-small-w8_3rdparty_in1k-256px --dest pretrained_ckpts
```

## ⚙️ 3. INSTANT Calibration
Our method requires custom calibration logic during the training loops. To enable this, you must replace the default `epoch_based_runner.py` in the installed `mmcv` library with our customized version.

1. Dynamically find the mmcv installation directory within the active Conda env
```bash
MMCV_DIR=$(python -c "import mmcv, os; print(os.path.dirname(mmcv.__file__))")
```
2. Define the path to the customized runner in this repository
```bash
CUSTOM_RUNNER="epoch_based_runner.py"
```
3. Create a backup of the original mmcv file (just in case)
```bash
cp "$MMCV_DIR/runner/epoch_based_runner.py" "$MMCV_DIR/runner/epoch_based_runner_backup.py"
```
4. Replace the original file with the customized version
```bash
cp "$CUSTOM_RUNNER" "$MMCV_DIR/runner/epoch_based_runner.py"
```
✅ Successfully patched epoch_based_runner.py in: $MMCV_DIR

## 🚀 4. Running the Code
Example of INSTANT:
```bash
# INSTANT, dataset CIFAR10, finetuning all layer (full), explained variance 0.95, oversampling 5
python train_INSTANT.py configs/efficientformer-l1/full_efficientformer-l1_cifar10.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth --compress --var 0.95 --calib_iter 5 --log-postfix instant_time --over_sam 5
```

Some common args:
- `--var`: The energy threshold use
- `--over_sampling`: The number of over sampling rank
- `--calib_iter`: Number of iterations used for once calibration

Follow `train_example.sh` to see scripts of other schemes (LBP-WHT, Vanilla, Gradient Filtering)