import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import os
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn as nn

from modules.MKSFA import MKSFANet
from utils.utils import PSNR, stitch_images
from utils.functions import AdversarialLoss, Discriminator, VGG16FeatureExtractor
from modules.FCR import FCR, CoherenceLoss


class Model():
    def __init__(self, args):
        self.lossNet = None
        self.iter = None
        self.l1_loss_val = 0.
        self.G_loss_val = 0.
        self.dataset = args.dataset

        self.summary = {}
        self.psnr = PSNR(255.0)
        self.mean_psnr = 0.
        self.board_writer = SummaryWriter(f"./logdir/")

        self.adv_loss = AdversarialLoss()
        self.coherence_loss = CoherenceLoss()
        self.fcr_loss = FCR(ablation=False)

    def initialize_model(self, path=None, train=True, finetune=False):
        self.G = MKSFANet()
        self.D = Discriminator(in_channels=3, use_sigmoid=True)
        if train:
            self.__train(finetune=finetune)
            self.G = nn.DataParallel(self.G, [0])
            self.D = nn.DataParallel(self.D, [0])
        self.optm_G = optim.Adam(self.G.parameters(), lr=2e-4)
        self.optm_D = optim.Adam(self.D.parameters(), lr=1e-5)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer', self.optm_G)])
            dirname, version = os.path.dirname(path), os.path.basename(path).split('_')[-1]
            load_ckpt(os.path.join(dirname, 'd_' + version), [('discriminator', self.D)], [('optimizer', self.optm_D)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=2e-5)
                self.optm_D = optim.Adam(self.D.parameters(), lr=1e-6)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            self.iter = 0

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
            self.G.cuda()
            self.D.cuda()
            self.psnr.cuda()
            self.adv_loss.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")

    def train(self, train_loader, save_path, iters=120000):
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        pbar = tqdm(total=iters)
        if self.iter > 0: pbar.update(self.iter)
        while self.iter < iters:
            for items in train_loader:

                real, mask = self.__cuda__(*items)
                masks = torch.cat([mask] * 3, dim=1)
                # gray_image = image_to_edge(real)   # []

                masked = real * masks  # [6,3,256,256]

                fake, comp = self.forward(masked, mask, real)
                self.run_discriminator_one_step(real, fake, comp, mask)
                self.run_generator_one_step(real, fake, comp, mask)

                self.iter += 1
                pbar.update(1)
                pbar.set_postfix(psnr=self.psnr_val)

                if self.iter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print(
                        "Iteration:%d, l1_loss:%.4f, psnr:%.4f, G_loss:%.4f, time_taken:%.2f" % (
                            self.iter, self.l1_loss_val / 50, self.mean_psnr / 50,
                            self.G_loss_val / 50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                    self.mean_psnr = 0.
                    self.G_loss_val = 0.

                if self.iter % 1000 == 0:
                    if self.iter % 10000 == 0:
                        if not os.path.exists('{:s}'.format(save_path)):
                            os.makedirs('{:s}'.format(save_path))
                        save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter), [('generator', self.G)],
                                  [('optimizer', self.optm_G)], self.iter)
                        save_ckpt('{:s}/d_{:d}.pth'.format(save_path, self.iter), [('discriminator', self.D)],
                                  [('optimizer', self.optm_D)], self.iter)
                        print('save model to ' + save_path)

                    images = stitch_images(
                        self.__postprocess(real),
                        self.__postprocess(masked),
                        self.__postprocess(comp)
                    )
                    if not os.path.exists('{:s}'.format(f"samples/{self.dataset}")):
                        os.makedirs('{:s}'.format(f"samples/{self.dataset}"))
                    samples_save_path = f"samples/{self.dataset}/{self.iter}.png"
                    images.save(samples_save_path)

                if self.iter >= iters: break

        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, "final"), [('generator', self.G)],
                      [('optimizer_G', self.optm_G)], self.iter)

    def test(self, test_loader, result_save_path, verbose):
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        pbar = tqdm(total=len(test_loader))
        for items in test_loader:
            items, files_name = items[:-1], items[-1]
            gt_images, mask = self.__cuda__(*items)

            masks = torch.cat([mask] * 3, dim=1)
            masked_images = gt_images * masks
            fake_B = self.G(masked_images)

            comp_B = fake_B * (1 - masks) + gt_images * masks
            os.makedirs('{:s}/inpainting/{}/'.format(result_save_path, self.dataset), exist_ok=True)

            for k in range(comp_B.size(0)):
                grid = make_grid(comp_B[k:k + 1])
                file_name_without_extension = os.path.splitext(files_name[k])[0]
                file_path = '{:s}/inpainting/{}/{}.png'.format(result_save_path, self.dataset,
                                                               file_name_without_extension)
                save_image(grid, file_path)

            count += 1
            pbar.update(1)

            if count >= 3000:
                break

    def forward(self, masked_image, mask, gt_images):
        fake_B = self.G(masked_image)
        masks = torch.cat([mask] * 3, dim=1)
        comp_B = fake_B * (1 - masks) + gt_images * masks

        return fake_B, comp_B

    def run_generator_one_step(self, gt_images, fake_images, comp_images, masks):
        self.optm_G.zero_grad()

        loss_G = self.get_g_loss(gt_images, fake_images, comp_images, masks)

        loss_G.backward()
        self.optm_G.step()

    def run_discriminator_one_step(self, gt_images, fake_images, comp_images, masks):
        self.optm_D.zero_grad()

        loss_D = self.get_d_loss(gt_images, fake_images)
        loss_D.backward()
        self.optm_D.step()

    def get_g_loss(self, real, fake, comp, mask):
        masks = torch.cat([mask] * 3, dim=1)
        masked = real * masks
        real_B, fake_B, comp_B = real, fake, comp

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats,
                                                                                                  comp_B_feats)
        l1_loss = self.l1_loss(real_B, fake_B)

        psnr = self.psnr(self.__postprocess(comp_B), self.__postprocess(real_B))
        self.psnr_val = psnr.item()
        self.mean_psnr += self.psnr_val

        fakes = fake_B
        gen_gan_feat = self.D(fakes)
        gen_gan_loss = self.adv_loss(gen_gan_feat, True, False)
        coherence_loss = self.coherence_loss(comp_B, real_B)
        fcr_loss = self.fcr_loss(comp_B, real_B, masked)

        self.__add_summary(self.board_writer, 'gan_loss/gen_adv', gen_gan_loss.item() * 0.001, self.iter)
        self.__add_summary(self.board_writer, 'loss/perceptual', preceptual_loss.item() * 0.05, self.iter)
        self.__add_summary(self.board_writer, 'loss/valid', l1_loss.item(), self.iter)
        self.__add_summary(self.board_writer, 'metric/psnr', psnr.item(), self.iter)
        self.__add_summary(self.board_writer, 'loss/coherence', coherence_loss.item() * 1, self.iter)
        self.__add_summary(self.board_writer, 'loss/fcr', fcr_loss.item() * 0.1, self.iter)

        loss_G = (
                + gen_gan_loss * 0.001
                + preceptual_loss * 0.05
                + l1_loss * 1
                + coherence_loss * 1
                + fcr_loss * 0.1
        )

        self.l1_loss_val += l1_loss.detach()
        self.G_loss_val += loss_G.detach()
        return loss_G

    def get_d_loss(self, real, fake):
        reals, fakes = real, fake

        real_feat = self.D(reals)
        fake_feat = self.D(fakes.detach())
        dis_real_loss = self.adv_loss(real_feat, True, True)
        dis_fake_loss = self.adv_loss(fake_feat, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        self.__add_summary(
            self.board_writer, 'gan_loss/dis_adv', dis_loss.item(), self.iter)

        return dis_loss

    def l1_loss(self, f1, f2):
        return torch.mean(torch.abs(f1 - f2))

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)

    def __add_summary(self, writer, name, val, iteration, prompt=False):
        INTERVAL = 10
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if prompt or (writer is not None and iteration % INTERVAL == 0):
            writer.add_scalar(name, self.summary[name] / INTERVAL, iteration)
            self.summary[name] = 0

    def __train(self, mode=True, finetune=False):
        if mode:
            super(MKSFANet, self.G).train(mode)
        if finetune:
            for name, module in self.G.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                    module.eval()

    def __postprocess(self, img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()