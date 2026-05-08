"""Encode FFHQ images into ALAE W-space latents and assign 6-class demographic labels.

Produces the NPZ files expected by drift_ffhq.py / train_drifting_ffhq.py:
    ffhq_latents_6class/train_latents_by_class.npz
    ffhq_latents_6class/test_latents_by_class.npz

Each NPZ has keys:
    male_children, male_adult, male_old,
    female_children, female_adult, female_old
each a float32 array of shape (N_c, 512).

Demographic classification uses FairFace (ResNet34), which predicts age group
and gender jointly. Age groups are collapsed to:
    children : age 0-2, 3-9, 10-19
    adult    : age 20-29, 30-39, 40-49
    old      : age 50-59, 60-69, 70+

This version expects the real FFHQ structure:

    ffhq-dataset-v2.json
    images1024x1024/
        00000/
            00000.png
            00001.png
            ...
        01000/
            ...

The official FFHQ JSON split is used for train/test instead of a random split.

Usage:
    python encode_latents_ffhq.py \
        --image-dir    /data/ffhq/images1024x1024 \
        --ffhq-json    /data/ffhq/ffhq-dataset-v2.json \
        --fairface-ckpt res34_fair_align_multi_7_20190809.pt \
        --alae-root    /path/to/ALAE \
        --out-dir      ffhq_latents_6class \
        --device       cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "male_children", "male_adult", "male_old",
    "female_children", "female_adult", "female_old",
]

FAIRFACE_AGE_GROUPS = [
    "0-2", "3-9", "10-19",
    "20-29", "30-39", "40-49",
    "50-59", "60-69", "70+",
]


def age_to_bucket(age_idx: int) -> str:
    label = FAIRFACE_AGE_GROUPS[age_idx]

    if label in ("0-2", "3-9", "10-19"):
        return "children"
    elif label in ("20-29", "30-39", "40-49"):
        return "adult"
    else:
        return "old"


def class_name(gender_idx: int, age_idx: int) -> str:
    gender = "male" if gender_idx == 0 else "female"
    return f"{gender}_{age_to_bucket(age_idx)}"


# ---------------------------------------------------------------------------
# FairFace classifier
# ---------------------------------------------------------------------------

class FairFaceClassifier(nn.Module):
    """
    FairFace ResNet34 classifier matching checkpoint keys exactly.
    No key replacement. Strict load only.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()

        self.model = tvm.resnet34(weights=None)
        self.model.fc = nn.Linear(512, 18)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        self.model.load_state_dict(state, strict=True)

        print(f"[fairface] strictly loaded FairFace ResNet34 from {ckpt_path}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        pred = logits.argmax(dim=1)

        age_idx = pred // 2
        gender_idx = pred % 2

        return age_idx, gender_idx
# ---------------------------------------------------------------------------
# Image dataset
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageFolderRecursive(Dataset):
    """Recursive FFHQ image tree. Returns (img_alae, img_fairface, path)."""

    ALAE_SIZE = 1024
    FAIRFACE_SIZE = 224

    def __init__(self, image_dir: str):
        image_dir_path = Path(image_dir).expanduser().resolve()

        self.paths = sorted(
            p for p in image_dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )

        if not self.paths:
            raise ValueError(f"No images found recursively under {image_dir_path}")

        print(f"[data] found {len(self.paths)} images under {image_dir_path}")

        self.tf_alae = transforms.Compose([
            transforms.Resize(
                self.ALAE_SIZE,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(self.ALAE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ])

        self.tf_ff = transforms.Compose([
            transforms.Resize(self.FAIRFACE_SIZE),
            transforms.CenterCrop(self.FAIRFACE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.tf_alae(img), self.tf_ff(img), str(img_path)


# ---------------------------------------------------------------------------
# FFHQ split loading
# ---------------------------------------------------------------------------

def load_ffhq_split_map(json_path: str) -> dict[str, str]:
    """
    Load official FFHQ train/test split information.

    Returns:
        dict mapping image stem to "train" or "test"

    Example:
        "00000" -> "train"
    """
    json_path_obj = Path(json_path).expanduser().resolve()

    with open(json_path_obj, "r") as f:
        meta = json.load(f)

    split_map: dict[str, str] = {}

    for _, entry in meta.items():
        image_info = entry.get("image", {})
        file_path = image_info.get("file_path")

        if file_path is None:
            continue

        stem = Path(file_path).stem
        category = str(entry.get("category", "")).lower()

        if category in {"training", "train"}:
            split_map[stem] = "train"
        elif category in {"validation", "valid", "val", "test"}:
            split_map[stem] = "test"

    if not split_map:
        raise ValueError(f"No usable train/test split entries found in {json_path_obj}")

    n_train = sum(v == "train" for v in split_map.values())
    n_test = sum(v == "test" for v in split_map.values())

    print(f"[ffhq-json] loaded {len(split_map)} split entries from {json_path_obj}")
    print(f"[ffhq-json] train={n_train}  test={n_test}")

    return split_map


# ---------------------------------------------------------------------------
# ALAE model loading
# ---------------------------------------------------------------------------

def load_alae(
    alae_root: str,
    alae_config: Optional[str],
    alae_artifacts: Optional[str],
    device: Optional[str] = "cuda"
) -> nn.Module:
    alae_root = str(Path(alae_root).expanduser().resolve())

    if alae_root not in sys.path:
        sys.path.insert(0, alae_root)

    try:
        from alae_ffhq_inference import load_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Could not import alae_ffhq_inference from {alae_root}. "
            f"Original error: {e}"
        )

    config = alae_config or os.path.join(alae_root, "configs", "ffhq.yaml")
    artifacts = alae_artifacts or os.path.join(
        alae_root,
        "training_artifacts",
        "ffhq",
    )

    cwd = os.getcwd()

    try:
        os.chdir(alae_root)
        model = load_model(config, training_artifacts_dir=artifacts)
    finally:
        os.chdir(cwd)

    return model.to(device)


@torch.no_grad()
def encode_alae(
    model: nn.Module,
    imgs_alae: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode [-1, 1] images to ALAE W-space latents [B, 512]."""
    model.cuda()
    imgs_alae = imgs_alae.to(device, non_blocking=True)

    out, _ = model.encode(
        imgs_alae,
        lod=model.layer_count - 1,
        blend_factor=1.0,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.ndim == 3:
        out = out.mean(dim=1)

    return out.cpu().float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode FFHQ to ALAE W-space latents plus FairFace labels."
    )

    p.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Root FFHQ image directory, e.g. images1024x1024.",
    )

    p.add_argument(
        "--ffhq-json",
        type=str,
        required=True,
        help="Path to official FFHQ JSON metadata, e.g. ffhq-dataset-v2.json.",
    )

    p.add_argument(
        "--fairface-ckpt",
        type=str,
        required=True,
        help="Path to FairFace ResNet34 checkpoint.",
    )

    p.add_argument(
        "--alae-root",
        type=str,
        default=os.environ.get("ALAE_ROOT", ""),
        help="Path to ALAE repo root. Can also set ALAE_ROOT env var.",
    )

    p.add_argument("--alae-config", type=str, default=None)
    p.add_argument("--alae-artifacts", type=str, default=None)

    p.add_argument(
        "--out-dir",
        type=str,
        default="ffhq_latents_6class",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ALAE encoding.",
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    if not args.alae_root:
        raise ValueError("Pass --alae-root /path/to/ALAE or set ALAE_ROOT env var.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load official FFHQ split
    # ------------------------------------------------------------------
    split_map = load_ffhq_split_map(args.ffhq_json)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = ImageFolderRecursive(args.image_dir)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("[alae] loading model ...")
    alae_model = load_alae(args.alae_root, args.alae_config, args.alae_artifacts)
    alae_model = alae_model.to(device).eval()

    for param in alae_model.parameters():
        param.requires_grad_(False)

    print("[alae] model loaded.")

    print(f"[fairface] loading classifier from {args.fairface_ckpt} ...")
    fairface = FairFaceClassifier(args.fairface_ckpt).to(device).eval()

    # ------------------------------------------------------------------
    # Encode and classify
    # ------------------------------------------------------------------
    all_latents: list[np.ndarray] = []
    all_class_names: list[str] = []
    all_stems: list[str] = []

    print("[encode] encoding images ...")

    for imgs_alae, imgs_ff, paths in tqdm(loader):
        latents = encode_alae(alae_model, imgs_alae, device)
        all_latents.append(latents.numpy())

        all_stems.extend([Path(p).stem for p in paths])

        with torch.no_grad():
            age_idx, gender_idx = fairface.predict(imgs_ff.to(device))

        for a, g in zip(age_idx.cpu().tolist(), gender_idx.cpu().tolist()):
            all_class_names.append(class_name(int(g), int(a)))

    all_latents_np = np.concatenate(all_latents, axis=0).astype(np.float32)

    print(f"[encode] done. latents shape: {all_latents_np.shape}")
    print(
        f"  mean={all_latents_np.mean():.4f}  "
        f"std={all_latents_np.std():.4f}  "
        f"min={all_latents_np.min():.4f}  "
        f"max={all_latents_np.max():.4f}"
    )

    # ------------------------------------------------------------------
    # Class indices
    # ------------------------------------------------------------------
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    class_indices = np.array(
        [name_to_idx[n] for n in all_class_names],
        dtype=np.int64,
    )

    print("[classify] FairFace class distribution:")

    for i, cname in enumerate(CLASS_NAMES):
        count = int((class_indices == i).sum())
        pct = 100 * count / len(class_indices)
        print(f"  {cname:<20s} {count:>6d}  ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Official FFHQ train/test split from JSON
    # ------------------------------------------------------------------
    train_mask = np.zeros(len(all_latents_np), dtype=bool)
    test_mask = np.zeros(len(all_latents_np), dtype=bool)

    missing: list[str] = []

    for i, stem in enumerate(all_stems):
        split = split_map.get(stem)

        if split == "train":
            train_mask[i] = True
        elif split == "test":
            test_mask[i] = True
        else:
            missing.append(stem)

    if missing:
        print(f"[warn] {len(missing)} images were missing from FFHQ JSON split map")
        print(f"[warn] first missing stems: {missing[:10]}")

    if train_mask.sum() == 0:
        raise ValueError("Official FFHQ JSON split produced 0 training images.")

    if test_mask.sum() == 0:
        raise ValueError("Official FFHQ JSON split produced 0 test images.")

    used_mask = train_mask | test_mask

    if not used_mask.all():
        dropped = int((~used_mask).sum())
        print(f"[warn] dropping {dropped} images without official split assignment")

    print(
        f"[split] train={train_mask.sum()}  "
        f"test={test_mask.sum()}  "
        f"source={args.ffhq_json}"
    )

    # ------------------------------------------------------------------
    # Save NPZs
    # ------------------------------------------------------------------
    for split_name, mask in [("train", train_mask), ("test", test_mask)]:
        split_latents = all_latents_np[mask]
        split_labels = class_indices[mask]

        payload: dict[str, np.ndarray] = {
            cname: split_latents[split_labels == c]
            for c, cname in enumerate(CLASS_NAMES)
        }

        npz_path = out_dir / f"{split_name}_latents_by_class.npz"
        np.savez_compressed(str(npz_path), **payload)

        print(f"[save] {split_name}: {npz_path}")

        for cname in CLASS_NAMES:
            print(f"  {cname:<20s} {payload[cname].shape[0]:>6d}")

    # ------------------------------------------------------------------
    # Flat debug arrays
    # ------------------------------------------------------------------
    np.save(str(out_dir / "all_latents.npy"), all_latents_np)
    np.save(str(out_dir / "all_class_indices.npy"), class_indices)
    np.save(str(out_dir / "train_mask.npy"), train_mask)
    np.save(str(out_dir / "test_mask.npy"), test_mask)
    np.save(str(out_dir / "used_mask.npy"), used_mask)
    np.save(str(out_dir / "all_stems.npy"), np.array(all_stems))

    print(f"[save] flat arrays saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()