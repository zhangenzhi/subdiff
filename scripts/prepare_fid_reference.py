"""
Prepare a reference image set for FID evaluation.

Samples N images from ImageNet val, applies the same eval transform used
during training (resize + center crop, NO normalization), saves as PNGs.

Usage:
  python scripts/prepare_fid_reference.py \
      --imagenet_dir /lustre1/work/c30636/dataset/imagenet \
      --num_images 5000 \
      --image_size 224 \
      --output_dir fid_reference/
"""

import os
import sys
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--imagenet_dir', type=str, required=True,
                   help='Root with train/ and val/ subdirs')
    p.add_argument('--num_images', type=int, default=5000)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # No normalization — FID needs RGB images in [0, 1] / [0, 255]
    transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 256 / 224)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),  # to [0, 1]
    ])

    data_dir = os.path.join(args.imagenet_dir, args.split)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print(f"Dataset size: {len(dataset)}")

    # Random subset
    indices = random.sample(range(len(dataset)), args.num_images)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    os.makedirs(args.output_dir, exist_ok=True)

    n_saved = 0
    for batch_idx, (imgs, _) in enumerate(loader):
        for i in range(imgs.shape[0]):
            img = imgs[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            path = os.path.join(args.output_dir, f'{n_saved:06d}.png')
            plt.imsave(path, img)
            n_saved += 1
        if (batch_idx + 1) % 10 == 0:
            print(f"  Saved {n_saved}/{args.num_images}")

    print(f"\nDone. Saved {n_saved} reference images to {args.output_dir}")


if __name__ == '__main__':
    main()
