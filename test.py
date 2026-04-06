"""
Offroad Segmentation — Mask2Former Inference Script
====================================================
Modes:
  --mode val   : run on labelled val set → prints per-class IoU + saves comparisons
  --mode test  : run on unlabelled testImages → saves predictions only

Usage:
  python test_segmentation.py --mode test
  python test_segmentation.py --mode val --output_dir ./predictions
"""

import os, warnings, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Class definitions — must match train_segmentation.py exactly
# ============================================================================
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}
N_CLASSES = 11
CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky",
]
COLOR_PALETTE = np.array([
    [0,   0,   0  ], [34,  139, 34 ], [0,   200, 0  ],
    [210, 180, 140], [139, 90,  43 ], [128, 128, 0  ],
    [255, 215, 0  ], [139, 69,  19 ], [128, 128, 128],
    [160, 82,  45 ], [135, 206, 235],
], dtype=np.uint8)

IMG_H = IMG_W = 512  # must match training

def convert_mask(pil):
    arr = np.array(pil)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[arr == raw] = cls
    return out

def colorize(mask_np):
    return COLOR_PALETTE[mask_np.astype(np.uint8)]

# ============================================================================
# Datasets
# ============================================================================
class ValDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.img_dir  = os.path.join(data_dir, "Color_Images")
        self.seg_dir  = os.path.join(data_dir, "Segmentation")
        self.ids      = sorted(os.listdir(self.img_dir))
        self.proc     = processor
        from torchvision.transforms.functional import resize as _resize
        self._resize  = _resize

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name  = self.ids[idx]
        image = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        image = image.resize((IMG_W, IMG_H), Image.BILINEAR)
        mask  = convert_mask(Image.open(os.path.join(self.seg_dir, name)))
        mask  = np.array(Image.fromarray(mask).resize((IMG_W, IMG_H), Image.NEAREST))
        inputs = self.proc(images=image, segmentation_maps=mask, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}, name


class TestDataset(Dataset):
    def __init__(self, data_dir, processor):
        sub = os.path.join(data_dir, "Color_Images")
        self.img_dir = sub if os.path.isdir(sub) else data_dir
        exts = {".png",".jpg",".jpeg",".bmp",".tiff"}
        self.ids  = sorted(f for f in os.listdir(self.img_dir)
                           if os.path.splitext(f)[1].lower() in exts)
        self.proc = processor

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name  = self.ids[idx]
        image = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        image = image.resize((IMG_W, IMG_H), Image.BILINEAR)
        inputs = self.proc(images=image, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}, name

# ============================================================================
# Metrics
# ============================================================================
def compute_iou_per_class(pred_flat, gt_flat, n=N_CLASSES):
    ious = []
    for c in range(n):
        inter = ((pred_flat==c) & (gt_flat==c)).sum().float()
        union = ((pred_flat==c) | (gt_flat==c)).sum().float()
        ious.append((inter/union).item() if union > 0 else float("nan"))
    return ious

def sem_logits_from_output(out, h, w):
    cq    = out.class_queries_logits            # (B,Q,C+1)
    mq    = out.masks_queries_logits            # (B,Q,h/4,w/4)
    mq_up = F.interpolate(mq, (h,w), mode="bilinear", align_corners=False)
    cp    = cq[..., :N_CLASSES].softmax(-1)
    mp    = mq_up.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", cp, mp)

def build_gt(batch, h, w, device):
    B  = len(batch["class_labels"])
    gt = torch.zeros(B, h, w, dtype=torch.long, device=device)
    for b in range(B):
        cls_l  = batch["class_labels"][b]
        msk_l  = batch["mask_labels"][b]
        if msk_l.numel() == 0:
            continue
        msk_up = F.interpolate(msk_l.float().unsqueeze(0),
                               size=(h,w), mode="nearest").squeeze(0).bool()
        for ci, mi in zip(cls_l, msk_up):
            gt[b][mi] = ci.long()
    return gt

def m2f_collate(batch):
    items, names = zip(*batch)
    out = {}
    for key in items[0]:
        vals = [d[key] for d in items]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out, list(names)

# ============================================================================
# Visualization helpers
# ============================================================================
def save_comparison(img_t, gt_np, pred_np, path, title=""):
    img = img_t.cpu().permute(1,2,0).numpy()
    img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots(1,3, figsize=(18,6))
    ax[0].imshow(img);            ax[0].set_title("Input");        ax[0].axis("off")
    ax[1].imshow(colorize(gt_np));  ax[1].set_title("Ground Truth"); ax[1].axis("off")
    ax[2].imshow(colorize(pred_np));ax[2].set_title("Prediction");   ax[2].axis("off")
    patches = [mpatches.Patch(color=COLOR_PALETTE[c]/255, label=CLASS_NAMES[c])
               for c in range(N_CLASSES)]
    ax[2].legend(handles=patches, bbox_to_anchor=(1.05,1), loc="upper left", fontsize=7)
    plt.suptitle(title); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def save_test_vis(img_t, pred_np, path, title=""):
    img = img_t.cpu().permute(1,2,0).numpy()
    img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].imshow(img);             ax[0].set_title("Input");      ax[0].axis("off")
    ax[1].imshow(colorize(pred_np));ax[1].set_title("Prediction"); ax[1].axis("off")
    patches = [mpatches.Patch(color=COLOR_PALETTE[c]/255, label=CLASS_NAMES[c])
               for c in range(N_CLASSES)]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05,1), loc="upper left", fontsize=7)
    plt.suptitle(title); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def save_metrics(mean_iou, per_class, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, "evaluation_metrics.txt")
    with open(fp, "w") as f:
        f.write("EVALUATION RESULTS\n" + "="*50 + "\n")
        f.write(f"Mean IoU : {mean_iou:.4f}\n" + "="*50 + "\n\n")
        f.write("Per-Class IoU:\n" + "-"*40 + "\n")
        for name, iou in zip(CLASS_NAMES, per_class):
            s = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<22}: {s}\n")
    print(f"Metrics -> {fp}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(12,6))
    valid = [v if not np.isnan(v) else 0 for v in per_class]
    ax.bar(range(N_CLASSES), valid, color=COLOR_PALETTE/255, edgecolor="black")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("IoU"); ax.set_title(f"Per-Class IoU  (Mean: {mean_iou:.4f})")
    ax.set_ylim(0,1); ax.axhline(mean_iou, color="red", linestyle="--", label="Mean")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_iou.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart  -> {out_dir}/per_class_iou.png")

# ============================================================================
# Main
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.path.join(script_dir, "mask2former_best.pth"))
    parser.add_argument("--mode", choices=["val","test"], default="test")
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--output_dir", default=os.path.join(script_dir, "predictions"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples",type=int, default=10)
    parser.add_argument("--model_ckpt", default="facebook/mask2former-swin-base-ade-semantic")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = (
            os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "val")
            if args.mode == "val" else
            os.path.join(script_dir, "..", "Offroad_Segmentation_testImages")
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Processor + model
    processor = Mask2FormerImageProcessor.from_pretrained(
        args.model_ckpt, do_resize=False, do_normalize=True,
        ignore_index=255, num_labels=N_CLASSES, reduce_labels=False,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_ckpt, num_labels=N_CLASSES, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval().to(device)
    print(f"Loaded weights from {args.model_path}")

    # Dataset
    if args.mode == "val":
        ds = ValDataset(args.data_dir, processor)
    else:
        ds = TestDataset(args.data_dir, processor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=m2f_collate)
    print(f"Dataset: {len(ds)} images")

    # Output dirs
    masks_dir   = os.path.join(args.output_dir, "masks")
    color_dir   = os.path.join(args.output_dir, "masks_color")
    vis_dir     = os.path.join(args.output_dir, "visualizations")
    for d in [masks_dir, color_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    all_class_ious, saved = [], 0

    with torch.no_grad():
        for batch, names in tqdm(loader, desc="Inferring", unit="batch"):
            pv = batch["pixel_values"].to(device)
            out = model(pixel_values=pv)
            h, w = pv.shape[-2:]
            sem  = sem_logits_from_output(out, h, w)
            preds = sem.argmax(1).cpu().numpy().astype(np.uint8)

            if args.mode == "val":
                gt = build_gt(batch, h, w, device)

            for i, name in enumerate(names):
                base = os.path.splitext(name)[0]
                p    = preds[i]

                # Save raw mask + colorized mask
                Image.fromarray(p).save(os.path.join(masks_dir, f"{base}_pred.png"))
                cv2.imwrite(os.path.join(color_dir, f"{base}_pred_color.png"),
                            cv2.cvtColor(colorize(p), cv2.COLOR_RGB2BGR))

                # Metrics (val only)
                if args.mode == "val":
                    gt_i = gt[i].cpu()
                    ious = compute_iou_per_class(
                        torch.tensor(p).view(-1), gt_i.view(-1))
                    all_class_ious.append(ious)

                # Visualizations
                if saved < args.num_samples:
                    if args.mode == "val":
                        save_comparison(pv[i], gt[i].cpu().numpy(), p,
                                        os.path.join(vis_dir, f"{base}_cmp.png"), name)
                    else:
                        save_test_vis(pv[i], p,
                                      os.path.join(vis_dir, f"{base}_vis.png"), name)
                    saved += 1

    if args.mode == "val" and all_class_ious:
        avg_class = np.nanmean(all_class_ious, axis=0)
        mean_iou  = float(np.nanmean(avg_class))
        print(f"\nMean IoU: {mean_iou:.4f}")
        for n_, iou in zip(CLASS_NAMES, avg_class):
            s = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            print(f"  {n_:<22}: {s}")
        save_metrics(mean_iou, avg_class, args.output_dir)

    print(f"\nDone!  Outputs -> {args.output_dir}/")
    print("  masks/          — raw class-ID masks")
    print("  masks_color/    — colorized masks")
    print("  visualizations/ — side-by-side comparisons")


if __name__ == "__main__":
    main()