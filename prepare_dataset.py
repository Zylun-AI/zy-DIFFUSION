import argparse
import os
import torch
import torchvision.transforms as T
from datasets import load_dataset
from tqdm import tqdm

def download_images(num_samples, img_size=256, outdir="data/text2img"):
    ds = load_dataset("jackyhate/text-to-image-2M", split="train", streaming=True)
    resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.LANCZOS)
    os.makedirs(outdir, exist_ok=True)
    infos = []
    n_ok = 0
    for i, sample in enumerate(tqdm(ds, total=num_samples, desc="downloading")):
        if n_ok >= num_samples:
            break
        try:
            img = sample["jpg"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = resize(img)
            fname = f"{n_ok:07d}.png"
            img.save(os.path.join(outdir, fname))
            prompt = sample["json"]["prompt"]
            infos.append({"file": fname, "prompt": prompt})
            n_ok += 1
        except Exception as e:
            pass
    # Salva prompts
    with open(os.path.join(outdir, "metadata.txt"), "w", encoding="utf-8") as f:
        for entry in infos:
            f.write(f'{entry["file"]}\t{entry["prompt"]}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--outdir", type=str, default="data/text2img")
    args = parser.parse_args()
    download_images(args.num_samples, args.img_size, args.outdir)