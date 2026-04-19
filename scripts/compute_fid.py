"""
Compute FID between two image directories.

Wraps `pytorch-fid` (pip install pytorch-fid) — uses InceptionV3 features.

Usage:
  # First time: pip install pytorch-fid

  python scripts/compute_fid.py \
      --reference fid_reference/ \
      --samples samples_naive_ddpm/ \
      --batch_size 50

  # Compare multiple methods at once:
  python scripts/compute_fid.py \
      --reference fid_reference/ \
      --samples samples_naive_ddpm/ samples_subdiff_dual/ samples_subdiff_orig/ \
      --batch_size 50
"""

import os
import sys
import argparse
import subprocess


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=str, required=True,
                   help='Reference image directory (real images)')
    p.add_argument('--samples', type=str, nargs='+', required=True,
                   help='One or more directories of generated samples')
    p.add_argument('--batch_size', type=int, default=50)
    p.add_argument('--device', type=str, default='cuda',
                   help='cuda or cpu')
    p.add_argument('--dims', type=int, default=2048,
                   help='Inception feature dim (2048 = standard FID, '
                        '768 / 192 / 64 also supported)')
    return p.parse_args()


def compute_fid(ref_dir, sample_dir, batch_size, device, dims):
    """Run pytorch-fid as a subprocess and parse the result."""
    cmd = [
        sys.executable, '-m', 'pytorch_fid',
        ref_dir, sample_dir,
        '--batch-size', str(batch_size),
        '--device', device,
        '--dims', str(dims),
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}")
        return None
    # Output format: "FID:  XX.XXXX"
    for line in result.stdout.split('\n'):
        if 'FID' in line:
            try:
                fid_val = float(line.split(':')[-1].strip())
                return fid_val
            except ValueError:
                pass
    print(f"  Could not parse FID from output:\n{result.stdout}")
    return None


def main():
    args = get_args()

    # Sanity checks
    if not os.path.isdir(args.reference):
        print(f"Reference dir not found: {args.reference}")
        sys.exit(1)
    n_ref = len([f for f in os.listdir(args.reference)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Reference set: {args.reference} ({n_ref} images)")

    results = {}
    for sample_dir in args.samples:
        if not os.path.isdir(sample_dir):
            print(f"Sample dir not found: {sample_dir}")
            continue
        n_sample = len([f for f in os.listdir(sample_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"\nMethod: {sample_dir} ({n_sample} images)")
        if n_sample < 100:
            print(f"  WARNING: only {n_sample} samples — FID needs at least "
                  f"~2000 for reliable estimate")

        fid = compute_fid(args.reference, sample_dir,
                          args.batch_size, args.device, args.dims)
        results[sample_dir] = fid

    # Final summary
    print("\n" + "=" * 60)
    print("FID Results")
    print("=" * 60)
    print(f"Reference: {args.reference} ({n_ref} images)")
    print()
    print(f"{'Method':<50} {'FID':>10}")
    print("-" * 62)
    for method, fid in sorted(results.items(), key=lambda x: x[1] or float('inf')):
        fid_str = f"{fid:.4f}" if fid is not None else "FAILED"
        print(f"{method:<50} {fid_str:>10}")


if __name__ == '__main__':
    main()
