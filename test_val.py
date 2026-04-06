import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ================= CONFIG =================
TEST_PATH = r"C:\Users\LENOVO\Downloads\Yolo\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages"
N_CLASSES = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CLASS MAP =================
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

def convert_mask(mask):
    mask = np.array(mask)
    out = np.zeros_like(mask)
    for k, v in VALUE_MAP.items():
        out[mask == k] = v
    return out

# ================= LOAD MODEL =================
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=N_CLASSES,
    ignore_mismatched_sizes=True,
    use_safetensors=True
).to(DEVICE)

model.load_state_dict(torch.load("segformer_epoch_13.pth"))
model.eval()

processor = SegformerImageProcessor()

# ================= DATA =================
img_dir = os.path.join(TEST_PATH, "Color_Images")
mask_dir = os.path.join(TEST_PATH, "Segmentation")

ids = os.listdir(img_dir)

ious_per_class = [[] for _ in range(N_CLASSES)]
all_correct = 0
all_pixels = 0

# ================= LOOP =================
for name in tqdm(ids):
    img = Image.open(os.path.join(img_dir, name)).convert("RGB")
    mask = Image.open(os.path.join(mask_dir, name))
    mask = convert_mask(mask)

    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inputs)

    pred = F.interpolate(
        out.logits,
        size=mask.shape,
        mode="bilinear",
        align_corners=False
    )
    pred = pred.argmax(1).squeeze().cpu().numpy()

    # Pixel Accuracy
    all_correct += (pred == mask).sum()
    all_pixels += mask.size

    # IoU per class
    for c in range(N_CLASSES):
        inter = np.logical_and(pred == c, mask == c).sum()
        union = np.logical_or(pred == c, mask == c).sum()
        if union > 0:
            iou = inter / union
            ious_per_class[c].append(iou)

# ================= METRICS =================

# Mean IoU
mean_ious = [np.mean(c) if len(c) > 0 else 0 for c in ious_per_class]
miou = np.mean(mean_ious)

# Pixel Accuracy
pixel_acc = all_correct / all_pixels

# ================= mAP@50 =================
# Count how many class predictions have IoU > 0.5

ap50_scores = []
for c in range(N_CLASSES):
    scores = ious_per_class[c]
    if len(scores) == 0:
        continue
    ap50 = np.mean([1 if s >= 0.5 else 0 for s in scores])
    ap50_scores.append(ap50)

map50 = np.mean(ap50_scores)

# ================= PRINT =================
print("\n===== TEST RESULTS =====")
print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"Mean IoU: {miou:.4f}")
print(f"mAP@50: {map50:.4f}")

print("\nPer-class IoU:")
for i, val in enumerate(mean_ious):
    print(f"{CLASS_NAMES[i]}: {val:.4f}")