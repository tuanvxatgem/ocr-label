"""
    Description: Create DataLoader for train, val, test
"""
import math
import os
from datetime import datetime
from operator import itemgetter

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_pad_images(img_h, img_w, images, keep_ratio_with_pad, vertical_lettering):
    # print([image.size for image in images])
    img_h_max = max(([image.size for image in images]), key=itemgetter(1))[1]
    img_w_max = max(([image.size for image in images]), key=itemgetter(0))[0]

    img_w_max = max(img_w_max, img_w)
    img_h_max = max(img_h_max, img_h)

    if keep_ratio_with_pad:
        input_channel = 3 if images[0].mode == 'RGB' else 1

        if vertical_lettering:
            # print("vertical_lettering", vertical_lettering)
            transform = NormalizePAD((input_channel, img_h_max, img_w), vertical_lettering)

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = h / float(w)
                if math.ceil(img_w * ratio) > img_h_max:
                    resized_h = img_h_max
                else:
                    resized_h = math.ceil(img_w * ratio)

                resized_image = image.resize((img_w, resized_h), Image.BICUBIC)
                resized_images.append(transform(resized_image))
        else:
            # same concept with 'Rosetta' paper

            resized_max_w = img_w_max
            transform = NormalizePAD((input_channel, img_h, resized_max_w), vertical_lettering)

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(img_h * ratio) > img_w_max:
                    resized_w = img_w_max
                else:
                    resized_w = math.ceil(img_h * ratio)

                resized_image = image.resize((resized_w, img_h), Image.BICUBIC)
                resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

    else:
        transform = ResizeNormalize((img_w_max, img_h))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

    return image_tensors


def log_error(exp_name, e, image_name=""):
    print(e)
    if not os.path.isfile(f'./saved_models/{exp_name}/log_errors.txt'):
        log = open(f'./saved_models/{exp_name}/log_errors.txt', "w")
    else:
        log = open(f'./saved_models/{exp_name}/log_errors.txt', "a")
    log.write(f"{datetime.now()}:{e}\t{image_name}\n")
    log.close()


class AlignCollate(object):
    def __init__(self, img_h=64, img_w=1000, keep_ratio_with_pad=False, vertical_lettering=False):
        self.imgH = img_h
        self.imgW = img_w
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.vertical_lettering = vertical_lettering

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        image_tensors = resize_pad_images(self.imgH, self.imgW, images, self.keep_ratio_with_pad,
                                          self.vertical_lettering)

        return image_tensors, labels


class ListDataset(Dataset):
    def __init__(self, list_img, opt):
        self.opt = opt
        self.list_img = list_img
        self.nSamples = len(self.list_img)
        self.list_hard_img = []

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if self.opt.rgb:
            img = Image.fromarray(np.uint8(self.list_img[index])).convert('RGB')
        else:
            print(index)
            img = Image.fromarray(np.uint8(self.list_img[index])).convert('L')
        return img, f"{index}"


class RawDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def get_gt(self, image_name):
        return ""

    def __getitem__(self, index):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        try:
            if self.opt.rgb:
                # for color image
                img = Image.open(f"{dir_name}/{self.image_folder}/{self.image_path_list[index]}").convert('RGB')
            else:
                img = Image.open(f"{dir_name}/{self.image_folder}/{self.image_path_list[index]}").convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return img, self.image_path_list[index]


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    def __init__(self, max_size, vertical_lettering, pad_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = pad_type
        self.vertical_lettering = vertical_lettering

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        if self.vertical_lettering:
            pad_img[:, :h, :] = img  # under pad
            if self.max_size[1] != h:  # add border Pad
                pad_img[:, h:, :] = img[:, h - 1, :].unsqueeze(1).expand(c, self.max_size[1] - h, w)
        else:
            pad_img[:, :, :w] = img  # right pad
            if self.max_size[2] != w:  # add border Pad
                pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return pad_img


def tensor2im(image_tensor, img_type=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(img_type)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
