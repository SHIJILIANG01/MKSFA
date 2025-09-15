import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from imageio import imread
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)
        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def random_crop(self, img):
        if not self.training: return img
        h, w = img.shape[:2]
        square_len = min(h, w)
        start_point = 0 if h == w else np.random.randint(low=0, high=max(h, w) - square_len)
        if h > w:
            img = img[start_point:start_point + square_len, ...]
        else:
            img = img[:, start_point:start_point + square_len, ...]
        return img

    def load_item(self, index):
        img = np.array(
            Image.fromarray(imread(self.data[index])).resize((self.target_size, self.target_size), Image.ANTIALIAS))

        if self.training:
            img = self.resize((img))
        else:
            img = self.resize((img), True)
        # load mask
        mask = self.load_mask(img, index)
        # augment data
        if self.training:
            if self.augment and np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]
            if self.augment and np.random.binomial(1, 0.5) > 0:
                mask = mask[:, ::-1, ...]
        if self.training:
            img = self.to_tensor(img)
            mask = self.to_tensor(mask)
            return img, mask
        else:  # external return value: file name
            img = self.to_tensor(img)
            mask = self.to_tensor(mask)
            return img, mask, self.data[index].split('/')[-1]

    def load_mask(self, img, index):

        # external mask, random order
        if self.mask_type == 0:
            # mask_index = index;
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = np.array(Image.fromarray(mask).convert('1').resize(size=(self.target_size, self.target_size)))
            mask = (mask > 0).astype(np.uint8) * 255
            mask = np.stack([mask] * 1, 2)
            if self.mask_reverse:
                return (255 - mask)
            else:
                return mask
        # generate random mask
        if self.mask_type == 1:
            if self.training:
                threshold = [0., 1.]
            else:
                threshold = [0.5, 0.6]  # mask ratio
            if np.random.binomial(1, 0.5) or (self.training == False):
                mask = 1 - generate_stroke_mask([self.target_size, self.target_size], threshold=threshold)
                mask = (mask > 0).astype(np.uint8) * 255
                mask = np.array(Image.fromarray(mask).resize(size=(self.target_size, self.target_size)))
            else:
                mask = 1 - generate_center_square_mask(self.target_size, self.training)
                mask = (mask > 0).astype(np.uint8) * 255
                mask = np.array(Image.fromarray(mask).resize(size=(self.target_size, self.target_size)))
            return mask

        # external mask, fixed order
        if self.mask_type == 2:
            mask_index = index
            mask = imread(self.mask_data[mask_index])
            mask = np.array(Image.fromarray(mask).convert('1').resize(size=(self.target_size, self.target_size)))
            mask = (mask > 0).astype(np.uint8)  # threshold due to interpolation
            mask = np.stack([mask] * 1, 2)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

        # square mask
        if self.mask_type == 3:
            mask = 1 - generate_center_square_mask(self.target_size, self.training)
            mask = (mask > 0).astype(np.uint8) * 255
            mask = np.array(Image.fromarray(mask).resize(size=(self.target_size, self.target_size)))
            return mask

    def resize(self, imgs, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        img = imgs
        imgh, imgw = img.shape[0:2]
        assert imgh / imgw
        if aspect_ratio_kept:
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                    # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]

                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]

        img = np.array(Image.fromarray(img).resize((self.target_size, self.target_size), Image.ANTIALIAS))

        return img

    def to_tensor(self, img):
        if img.shape[2] == 3:
            img = Image.fromarray(img)
            img_t = F.to_tensor(img).float()
        else:
            img = np.copy(img)
            img_t = torch.from_numpy(img.transpose((2, 0, 1)))
            if isinstance(img_t, torch.ByteTensor):
                img_t = img_t.float().div(255)
        return img_t

    def load_list(self, path):
        if isinstance(path, str):
            if os.path.isdir(path):
                path = list(glob.glob(path + '/*.jpg')) + list(glob.glob(path + '/*.png')) + list(
                    glob.glob(path + '/*.npy'))
                path.sort()
                return path
            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []


def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360,
                         threshold=[0.1, 0.2]):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(max_parts // 5, max_parts) if threshold[1] == 1 else 1000
    upper_bound, lower_bound = int(threshold[1] * im_size[0] * im_size[1]), int(threshold[0] * im_size[0] * im_size[1])
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        if threshold[1] != 1.:
            if mask.sum() > upper_bound:
                break
            elif mask.sum() > lower_bound and np.random.binomial(1, 0.8):
                break
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_center_square_mask(target_size, is_training):
    if is_training:
        crop_seed = np.random.uniform(low=2, high=3)
    else:
        crop_seed = 2
    crop_h = crop_w = int(target_size // crop_seed)
    # crop_h = crop_w =  96   # only for CityScape dataset
    # dis = 80                # only for CityScape dataset
    center_crop = np.ones((crop_h, crop_w, 1))
    mask = np.zeros((target_size, target_size, 1))
    dis = target_size // 4  # 边距
    if is_training: dis += np.random.randint(low=0, high=target_size // 4)
    mask[dis: dis + crop_h, dis: dis + crop_w, :] = center_crop
    return np.concatenate([mask, mask, mask], axis=2)