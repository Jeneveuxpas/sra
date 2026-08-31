import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import argparse
import os
import tempfile
import zipfile

def create_npz_from_sample_folder(sample_dir, num=50_000,save_path=None):
    """
    Build an ADM-compatible ``arr_0`` NPZ without materializing all images in RAM.

    A 50K ImageNet-256 array is about 9.8 GiB. ``np.stack`` followed by
    ``np.savez`` transiently needs multiple copies of that array and can swap
    indefinitely. We instead write a temporary NPY memmap one image at a time,
    then add that NPY file to an uncompressed ZIP container (the NPZ format).
    """
    npz_path = save_path if save_path else f"{sample_dir}.npz"
    output_dir = os.path.dirname(os.path.abspath(npz_path)) or "."
    with Image.open(f"{sample_dir}/000000.png") as first_image:
        first = np.asarray(first_image.convert("RGB"), dtype=np.uint8)
    height, width, channels = first.shape
    expected_shape = (num, height, width, channels)

    fd, tmp_npy_path = tempfile.mkstemp(
        prefix=".npz_convert_", suffix=".npy", dir=output_dir
    )
    os.close(fd)
    tmp_npz_path = f"{npz_path}.tmp"
    try:
        samples = np.lib.format.open_memmap(
            tmp_npy_path, mode="w+", dtype=np.uint8, shape=expected_shape
        )
        for i in tqdm(range(num), desc="Writing images to temporary NPY"):
            with Image.open(f"{sample_dir}/{i:06d}.png") as sample_pil:
                sample = np.asarray(sample_pil.convert("RGB"), dtype=np.uint8)
            if sample.shape != (height, width, channels):
                raise ValueError(
                    f"Sample {i:06d} has shape {sample.shape}; expected "
                    f"{(height, width, channels)}"
                )
            samples[i] = sample
        samples.flush()
        del samples

        print("Packing NPZ from temporary NPY (disk streaming, no large RAM allocation)...")
        with zipfile.ZipFile(tmp_npz_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(tmp_npy_path, arcname="arr_0.npy")
        os.replace(tmp_npz_path, npz_path)
    finally:
        if os.path.exists(tmp_npy_path):
            os.remove(tmp_npy_path)
        if os.path.exists(tmp_npz_path):
            os.remove(tmp_npz_path)

    print(f"Saved .npz file to {npz_path} [shape={expected_shape}].")
    return npz_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, help="Path to the directory containing .png samples.")
    parser.add_argument("--num-fid-samples", type=int, default=50_000, help="Number of samples to convert.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="sde")
    args = parser.parse_args()

    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    folder_name = f"{exp_name}-{ckpt_string_name}-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    save_path = f"{args.sample_dir}/{folder_name}.npz"
    print("sample_npz_save_path:",save_path)
    create_npz_from_sample_folder(sample_folder_dir, num=args.num_fid_samples,save_path=save_path)
