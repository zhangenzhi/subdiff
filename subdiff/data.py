"""
ImageNet data loading for SubDiff pretraining and linear probe evaluation.

Two backends:
  - 'torch'  (default): standard torchvision ImageFolder + DataLoader.
  - 'dali': NVIDIA DALI pipeline with GPU-side JPEG decode + augmentations.
            Requires `pip install nvidia-dali-cuda120` (or matching CUDA).
            Falls back to torch with a warning if DALI is not importable.

DALI is the right choice when:
  - JPEG decode is the bottleneck (CPU-bound dataloader).
  - The filesystem (e.g. lustre) handles sequential reads well per worker.

It is NOT a fix for "lustre random small-file IO" itself — for that,
re-shard ImageNet into tar/webdataset/ffcv format. DALI just removes the
CPU decode overhead which often hides the IO problem.
"""

import os
import warnings
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Standard torchvision pipeline
# ---------------------------------------------------------------------------

def build_pretrain_transform(image_size=224):
    """SSL-style augmentation: aggressive RandomResizedCrop (scale 0.2-1.0)
    teaches the encoder to recognize local crops of natural scenes. Appropriate
    for representation learning (MAE/DINO/SimCLR) but TOO AGGRESSIVE for
    generative models, which should see full-scene distributions."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_diffusion_transform(image_size=224):
    """Standard generation-training augmentation (DDPM/DiT style):
    - Resize short side to 256
    - Center crop to image_size (deterministic, no aggressive scaling)
    - Horizontal flip only
    - ImageNet normalize
    Preserves the natural-scene distribution so model learns to generate
    coherent full images, not local texture crops."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_torch_pretrain(imagenet_dir, image_size, batch_size, num_workers,
                          distributed, transform_type='ssl'):
    train_dir = os.path.join(imagenet_dir, 'train')
    if transform_type == 'diffusion':
        transform = build_diffusion_transform(image_size)
    else:
        transform = build_pretrain_transform(image_size)
    dataset = datasets.ImageFolder(train_dir, transform=transform)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler


def _build_torch_eval(imagenet_dir, image_size, batch_size, num_workers, split, distributed):
    data_dir = os.path.join(imagenet_dir, split)
    transform = build_eval_transform(image_size)
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# DALI pipeline (GPU JPEG decode + GPU augmentations)
# ---------------------------------------------------------------------------

def _try_import_dali():
    try:
        from nvidia.dali import pipeline_def, fn, types
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
        return pipeline_def, fn, types, DALIClassificationIterator, LastBatchPolicy
    except ImportError:
        return None


class _DALIWrapper:
    """Adapt DALIClassificationIterator into a (imgs, labels) iterable that
    looks like torch DataLoader. Also exposes len() and a no-op set_epoch()."""

    def __init__(self, dali_iter, num_batches):
        self.dali_iter = dali_iter
        self.num_batches = num_batches

    def __iter__(self):
        for batch in self.dali_iter:
            data = batch[0]
            yield data['data'], data['label'].squeeze(-1).long()
        self.dali_iter.reset()

    def __len__(self):
        return self.num_batches


class _DALISampler:
    """Stub that satisfies sampler.set_epoch(epoch) calls. DALI handles
    shuffling internally; we just need the API."""

    def set_epoch(self, epoch):
        pass


def _build_dali_pretrain(imagenet_dir, image_size, batch_size, num_workers, distributed):
    dali_imports = _try_import_dali()
    if dali_imports is None:
        warnings.warn("nvidia-dali not installed; falling back to torch dataloader. "
                      "Install with `pip install nvidia-dali-cuda120`.")
        return _build_torch_pretrain(imagenet_dir, image_size, batch_size, num_workers, distributed)

    pipeline_def, fn, types, DALIClassificationIterator, LastBatchPolicy = dali_imports

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    train_dir = os.path.join(imagenet_dir, 'train')

    # Count files for length (DALI doesn't report dataset_size up front)
    # Use ImageFolder's class scanning logic — fast since it only does directory walk.
    n_total = sum(len(os.listdir(os.path.join(train_dir, c)))
                  for c in os.listdir(train_dir))
    num_batches = (n_total // world_size) // batch_size

    @pipeline_def(batch_size=batch_size, num_threads=num_workers, device_id=local_rank)
    def train_pipeline():
        # File reader with built-in shuffling and DDP-aware sharding
        jpegs, labels = fn.readers.file(
            file_root=train_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=True,
            initial_fill=2048,
            name='Reader',
        )
        # JPEG decode on GPU (mixed: header on CPU, decode on GPU)
        images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
        # RandomResizedCrop equivalent
        images = fn.random_resized_crop(
            images, size=image_size,
            random_area=[0.2, 1.0],
            random_aspect_ratio=[0.75, 1.333],
        )
        # Random horizontal flip
        images = fn.flip(images, horizontal=fn.random.coin_flip())
        # Normalize + HWC -> CHW
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout='CHW',
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        return images, labels.gpu()

    pipe = train_pipeline()
    pipe.build()

    dali_iter = DALIClassificationIterator(
        pipe, reader_name='Reader',
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=False,
    )
    return _DALIWrapper(dali_iter, num_batches), _DALISampler()


def _build_dali_eval(imagenet_dir, image_size, batch_size, num_workers, split, distributed):
    dali_imports = _try_import_dali()
    if dali_imports is None:
        warnings.warn("nvidia-dali not installed; falling back to torch eval dataloader.")
        return _build_torch_eval(imagenet_dir, image_size, batch_size, num_workers, split, distributed)

    pipeline_def, fn, types, DALIClassificationIterator, LastBatchPolicy = dali_imports

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    data_dir = os.path.join(imagenet_dir, split)
    n_total = sum(len(os.listdir(os.path.join(data_dir, c)))
                  for c in os.listdir(data_dir))
    num_batches = (n_total // world_size + batch_size - 1) // batch_size

    resize = int(image_size * 256 / 224)

    @pipeline_def(batch_size=batch_size, num_threads=num_workers, device_id=local_rank)
    def eval_pipeline():
        jpegs, labels = fn.readers.file(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=False,
            name='Reader',
        )
        images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
        images = fn.resize(images, resize_shorter=resize)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout='CHW',
            crop=[image_size, image_size],
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        return images, labels.gpu()

    pipe = eval_pipeline()
    pipe.build()

    dali_iter = DALIClassificationIterator(
        pipe, reader_name='Reader',
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=False,
    )
    return _DALIWrapper(dali_iter, num_batches), _DALISampler()


# ---------------------------------------------------------------------------
# Public dispatchers — backend selected by `backend` arg
# ---------------------------------------------------------------------------

def build_pretrain_dataloader(imagenet_dir, image_size=224, batch_size=256,
                              num_workers=8, distributed=False, backend='torch',
                              transform_type='ssl'):
    """transform_type:
      - 'ssl' (default): aggressive RandomResizedCrop for representation learning
      - 'diffusion': center-crop + flip only, preserves natural-scene distribution
    """
    if backend == 'dali':
        return _build_dali_pretrain(imagenet_dir, image_size, batch_size,
                                    num_workers, distributed)
    return _build_torch_pretrain(imagenet_dir, image_size, batch_size,
                                 num_workers, distributed, transform_type)


def build_eval_dataloader(imagenet_dir, image_size=224, batch_size=256,
                          num_workers=8, split='val', distributed=False, backend='torch'):
    if backend == 'dali':
        return _build_dali_eval(imagenet_dir, image_size, batch_size,
                                num_workers, split, distributed)
    return _build_torch_eval(imagenet_dir, image_size, batch_size,
                             num_workers, split, distributed)
