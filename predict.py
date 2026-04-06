import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=11,
    ignore_mismatched_sizes=True,
    use_safetensors=True
).to(device)

model.load_state_dict(torch.load("best_model.pth"))  # or your saved file
model.eval()

processor = SegformerImageProcessor()

# ================= COLOR MAP =================
COLOR_MAP = np.array([
    [0, 0, 0],          # Background
    [34, 139, 34],     # Trees
    [0, 200, 0],       # Lush Bushes
    [210, 180, 140],   # Dry Grass
    [139, 90, 43],     # Dry Bushes
    [128, 128, 0],     # Ground Clutter
    [255, 215, 0],     # Flowers
    [139, 69, 19],     # Logs
    [128, 128, 128],   # Rocks
    [160, 82, 45],     # Landscape
    [135, 206, 235],   # Sky
], dtype=np.uint8)

# ================= LOAD IMAGE =================
img_path = "img2.png"   # 👈 CHANGE THIS
img = Image.open(img_path).convert("RGB")

# ================= PREPROCESS =================
inputs = processor(images=img, return_tensors="pt").to(device)

# ================= PREDICT =================
with torch.no_grad():
    out = model(**inputs)

# Resize to original image size
pred = F.interpolate(
    out.logits,
    size=img.size[::-1],
    mode="bilinear",
    align_corners=False
)

pred = pred.argmax(1).squeeze().cpu().numpy()

# ================= COLORIZE =================
color_pred = COLOR_MAP[pred]

# ================= SAVE =================
output_path = "prediction.png"
Image.fromarray(color_pred).save(output_path)

print(f"✅ Prediction saved at: {output_path}")