# Prerequisites

 1. Python 3
 2. Pytorch 2.0
 3. NVIDIA GPU + CUDA cuDNN

# Run
1. train the model
```bash
python train.py --data_root ../training_data/ --mask_root your_mask_dir
```
2. test the model
```bash
python test.py --data_root ../training_data/ --mask_root your_mask_dir
```

# Download Dataset
we use [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ), [FFHQ](https://github.com/NVlabs/ffhq-dataset), and [Paris StreetView datasets](https://github.com/pathak22/context-encoder). [Liu et al.](https://arxiv.org/abs/1804.07723) provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting) as the training and testing mask.
