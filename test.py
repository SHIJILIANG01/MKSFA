import os
import argparse
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader


def set_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebAMask-HQ',
                        help='dataset name. choice of {CelebAMask-HQ, FFHQ, Paris}')
    parser.add_argument('--data_root', type=str, default='../training_data/')
    parser.add_argument('--mask_root', type=str, default='your_mask_dir', help='optional')
    parser.add_argument('--save_root', type=str, default='./result', help='for test results')
    parser.add_argument('--mask_mode', type=int, default=2)
    parser.add_argument('--g_path', type=str, default="./checkpoint/CelebAMask-HQ/g_240000.pth",
                        help='model path when testing')
    parser.add_argument('--d_path', type=str, default="./checkpoint/CelebAMask-HQ/d_240000.pth",
                        help='model path when testing')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    args.img_root = os.path.join(args.data_root + args.dataset + '/img/test/')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = Model(args)
    set_seed(2233)
    model.initialize_model(args.g_path, args.d_path, False)
    model.cuda()
    dataset = Dataset(args.img_root, args.mask_root, args.mask_mode, target_size=256, mask_reverse=True, training=False)
    dataloader = DataLoader(dataset)
    model.test(dataloader, args.save_root, args.verbose)


if __name__ == '__main__':
    run()