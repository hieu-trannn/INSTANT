source ~/miniconda3/etc/profile.d/conda.sh 
conda activate instant

for ds in cifar10 cifar100
do
    for ord in 2 4 8
    do
        echo Train EfficientFormer-L1 on $ds with LP_L1-$ord
        python train_LBP_WHT.py configs/efficientformer-l1/filt_lptri_full_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth --cfg-options gradient_filter.common_cfg.extra_cfg.lp_tri_order=$ord --log-postfix lbp_wht_ord$ord
    done
    
    for ord in 0 5 7
    do
        echo Train EfficientFormer-L1 on $ds with INSTANT-var_0.95-oversampling-$ord
        python train_INSTANT.py configs/efficientformer-l1/full_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth --compress --var 0.95 --calib_iter 5 --log-postfix instant_time --over_sam $ord --log-postfix instant_oversamp_$ord
    done

    echo Train EfficientFormer-L1 on $ds with Vanilla
    python train_INSTANT.py configs/efficientformer-l1/full_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth --log-postfix vanilla

    echo Train EfficientFormer-L1 on $ds with Gradient Filtering
    python train_GF.py configs/efficientformer-l1/filt_lptri_full_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth --cfg-options gradient_filter.common_cfg.extra_cfg.lp_tri_order=2 --log-postfix gradient_filtering

done