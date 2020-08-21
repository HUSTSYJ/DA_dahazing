# DA_dahazing
This is the PyTorch implementation for our CVPR'20 paper:

**Yuanjie Shao, Lerenhan Li, Wenqi Ren, Changxin Gao, Nong Sang. Domain Adaptation for Image Dehazing. [PAPER](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shao_Domain_Adaptation_for_Image_Dehazing_CVPR_2020_paper.pdf) **

## Environment
1. Python 3.6
2. PyTorch 1.0.0
3. CUDA 9.2
4. Ubuntu 16.04

## Training 
- Dataset
Google drive: [DATASETS](https://drive.google.com/drive/folders/10cP6Z-n2G0006_ppW1WxkQpNKg3mSfnj?usp=sharing).

- Train CycleGAN 
```
python train.py --dataroot ./datasets/dehazing --name run_cyclegan --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8091 --which_model_netG resnet_9blocks --lambda_A 1 --lambda_B 1 --lambda_identity 0.1   --niter 90 --niter_decay 0 --fineSize 256 --no_html --batchSize 2  --gpu_id 2 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model cyclegan
```

- Train Fr using the pretrained CycleGAN
```
python train.py  --dataroot ./datasets/dehazing --name run_fr_depth --lambda_Dehazing 10 --lambda_Dehazing_DC 1e-2 --lambda_Dehazing_TV 1e-2 --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8090  --epoch_count 1 --niter 90 --niter_decay 0 --fineSize 256 --no_html --batchSize 2   --gpu_id 3 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model RDehazingnet --g_s2r_premodel ./checkpoints/run_cyclegan/netG_A.pth  
```

- Train Fs using the pretrained CycleGAN
```
python train.py  --dataroot ./datasets/dehazing --name run_fs_depth --lambda_Dehazing 10 --lambda_Dehazing_DC 1e-2 --lambda_Dehazing_TV 1e-2 --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8094  --epoch_count 1 --niter 90 --niter_decay 0 --fineSize 256 --no_html --batchSize 2   --gpu_id 3 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model SDehazingnet --g_r2s_premodel ./checkpoints/run_cyclegan/netG_B.pth 
```

- Train DA_dehazing using the pretrained Fr, Fs and CycleGAN.
```
python train.py  --dataroot ./datasets/dehazing --name run_danet_depth --epoch_count 1 --niter 50 --lambda_S 1 --lambda_R 1 --lambda_identity 0.1 --lambda_Dehazing 10 --lambda_Dehazing_Con 0.1 --lambda_Dehazing_DC 1e-2 --lambda_Dehazing_TV 1e-3 --learn_residual --resize_or_crop crop --display_freq 100 --print_freq 100 --display_port 8094 --niter_decay 0 --fineSize 256 --no_html --batchSize 2   --gpu_id 3 --update_ratio 1 --unlabel_decay 0.99 --save_epoch_freq 1 --model danet --S_Dehazing_premodel ./checkpoints/run_fs_depth/netS_Dehazing.pth --R_Dehazing_premodel ./checkpoints/run_fr_depth/netR_Dehazing.pth --g_s2r_premodel ./checkpoints/run_cyclegan_depth/netG_A.pth --g_r2s_premodel ./checkpoints/run_cyclegan/netG_B.pth --d_r_premodel ./checkpoints/run_cyclegan/netD_A.pth --d_s_premodel ./checkpoints/run_cyclegan/netD_B.pth
```


## Test
Baidu network disk： [MODELS](https://pan.baidu.com/s/1AYswMVKk-rX0OkTS9pzNkg).
Extraction code：8326

Google drive: [MODELS](https://drive.google.com/file/d/1jQv_IVLHO98Nuj-wS0ebHE1GI1uqYokS/view?usp=sharing).

```
python test.py --dataroot ./datasets/dehazing --name run_test --learn_residual --resize_or_crop crop --display_port 8095 --which_model_netG resnet_9blocks  --batchSize 1 --gpu_id 3 --model SDehazingnet --S_Dehazing_premodel ./checkpoints/30_netS_Dehazing.pth
```

```
python test.py --dataroot ./datasets/dehazing --name run_test --learn_residual --resize_or_crop crop --display_port 8095 --which_model_netG resnet_9blocks  --batchSize 1 --gpu_id 3 --model RDehazingnet --R_Dehazing_premodel ./checkpoints/30_netR_Dehazing.pth
```
 
## Acknowledgments
Code is inspired by [GASDA](https://github.com/sshan-zhao/GASDA) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Contact
Yuanjie Shao: shaoyuanjie@hust.edu.cn
