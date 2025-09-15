# Prerequisites

 1. Python 3
 2. Pytorch 2.0
 3. NVIDIA GPU + CUDA cuDNN

# Run
1. train the model
python train.py --dataroot no_use --name celebahq_LGNet --model pix2pixglg --netG1 unet_256 --netG2 resnet_4blocks --netG3 unet256 --netD snpatch --gan_mode lsgan --input_nc 4 --no_dropout --direction AtoB --display_id 0 --gpu_ids 0
3. test the model
python test_and_save.py --dataroot no_use --name celebahq_LGNet --model pix2pixglg --netG1 unet_256 --netG2 resnet_4blocks --netG3 unet256 --gan_mode nogan --input_nc 4 --no_dropout --direction AtoB --gpu_ids 0
# Download Dataset
We use
