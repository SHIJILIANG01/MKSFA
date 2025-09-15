import os
import argparse
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebAMask-HQ',
                        help='dataset name. set of {CelebAMask-HQ, FFHQ, Paris}')
    parser.add_argument('--data_root', type=str, default='../training_data/')
    parser.add_argument('--mask_root', type=str, default='your_mask_dir')
    parser.add_argument('--mask_mode', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=240000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=2)
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()

    args.img_root = os.path.join(args.data_root + args.dataset + '/img/train/')

    args.model_save_path = os.path.join('checkpoint/', args.dataset)
    args.resume_ckpt = f'checkpoint/{args.dataset}/g_{args.num_iters}.pth'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = Model(args)
    # pretrain
    model.initialize_model('', True, False)
    model.cuda()
    dataset = Dataset(args.img_root, args.mask_root, args.mask_mode, target_size=256, mask_reverse=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    model.train(dataloader, args.model_save_path, args.num_iters)
    # fintune
    model = Model(args)
    model.initialize_model(args.resume_ckpt, True, True)
    model.cuda()
    dataset = Dataset(args.img_root, args.mask_root, args.mask_mode, target_size=256, mask_reverse=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    model.train(dataloader, args.model_save_path, args.num_iters * 1)


if __name__ == '__main__':
    run()