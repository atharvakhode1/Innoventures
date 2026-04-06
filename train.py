# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm
# from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# # ================= CONFIG =================
# CFG = {
#     "model_ckpt": "nvidia/segformer-b2-finetuned-ade-512-512",
#     "img_size": 512,
#     "batch_size": 2,
#     "lr": 5e-5,
#     "epochs": 20,
# }

# # ================= CLASSES =================
# VALUE_MAP = {
#     0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
#     550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
# }
# N_CLASSES = 11

# def convert_mask(mask):
#     mask = np.array(mask)
#     out = np.zeros_like(mask)
#     for k, v in VALUE_MAP.items():
#         out[mask == k] = v
#     return out

# # ================= DATASET =================
# class DesertDataset(Dataset):
#     def __init__(self, root, processor):
#         self.img_dir = os.path.join(root, "Color_Images")
#         self.mask_dir = os.path.join(root, "Segmentation")
#         self.ids = os.listdir(self.img_dir)
#         self.processor = processor

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         name = self.ids[idx]

#         img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
#         mask = Image.open(os.path.join(self.mask_dir, name))
#         mask = convert_mask(mask)

#         inputs = self.processor(
#             images=img,
#             segmentation_maps=mask,
#             return_tensors="pt"
#         )

#         return {
#             "pixel_values": inputs["pixel_values"].squeeze(0),
#             "labels": inputs["labels"].squeeze(0)
#         }

# # ================= METRIC =================
# def compute_miou(pred, label):
#     pred = pred.view(-1)
#     label = label.view(-1)

#     ious = []
#     for c in range(N_CLASSES):
#         inter = ((pred == c) & (label == c)).sum().float()
#         union = ((pred == c) | (label == c)).sum().float()
#         if union > 0:
#             ious.append((inter / union).item())

#     return np.mean(ious) if ious else 0
#     best_iou = 0
#     if val_iou > best_iou:
#     best_iou = val_iou
#     torch.save(model.state_dict(), "best_model.pth")
#     print("✅ Best model saved!")

# # ================= TRAIN =================
# def train():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     processor = SegformerImageProcessor()
#     model = SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b2-finetuned-ade-512-512",
#     num_labels=N_CLASSES,
#     ignore_mismatched_sizes=True,
#     use_safetensors=True   # 👈 THIS LINE IS MANDATORY
#     ).to(device)

#     root = r"C:\Users\LENOVO\Downloads\Yolo\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"

#     train_ds = DesertDataset(os.path.join(root, "train"), processor)
#     val_ds   = DesertDataset(os.path.join(root, "val"), processor)

#     train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"])

#     optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])

#     try:    
#         for epoch in range(CFG["epochs"]):
#             model.train()
#             total_loss = 0

#             for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#                 pixel = batch["pixel_values"].to(device)
#                 labels = batch["labels"].to(device)

#                 out = model(pixel_values=pixel, labels=labels)
#                 loss = out.loss

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#             print(f"Train Loss: {total_loss/len(train_loader):.4f}")

#             # ===== VALIDATION =====
#             model.eval()
#             ious = []

#             with torch.no_grad():
#                 for batch in val_loader:
#                     pixel = batch["pixel_values"].to(device)
#                     labels = batch["labels"].to(device)

#                     out = model(pixel_values=pixel)
#                     preds = F.interpolate(out.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
#                     preds = preds.argmax(1)

#                     ious.append(compute_miou(preds, labels))

#             print(f"Val mIoU: {np.mean(ious):.4f}")

#         torch.save(model.state_dict(), "segformer_b2.pth")
#         print("Model saved!")

#     except KeyboardInterrupt:
#     torch.save(model.state_dict(), "interrupted_model.pth")
#     print("⚠️ Model saved after interruption")

# if __name__ == "__main__":
#     train()

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ================= CONFIG =================
CFG = {
    "model_ckpt": "nvidia/segformer-b2-finetuned-ade-512-512",
    "img_size": 512,
    "batch_size": 2,
    "lr": 5e-5,
    "epochs": 20,
}

# ================= CLASSES =================
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}
N_CLASSES = 11

def convert_mask(mask):
    mask = np.array(mask)
    out = np.zeros_like(mask)
    for k, v in VALUE_MAP.items():
        out[mask == k] = v
    return out

# ================= DATASET =================
class DesertDataset(Dataset):
    def __init__(self, root, processor):
        self.img_dir = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")
        self.ids = os.listdir(self.img_dir)
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        mask = convert_mask(mask)

        inputs = self.processor(
            images=img,
            segmentation_maps=mask,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["labels"].squeeze(0)
        }

# ================= METRIC =================
def compute_miou(pred, label):
    pred = pred.view(-1)
    label = label.view(-1)

    ious = []
    for c in range(N_CLASSES):
        inter = ((pred == c) & (label == c)).sum().float()
        union = ((pred == c) | (label == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())

    return np.mean(ious) if ious else 0


# ================= TRAIN =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = SegformerImageProcessor()

    model = SegformerForSemanticSegmentation.from_pretrained(
        CFG["model_ckpt"],
        num_labels=N_CLASSES,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    ).to(device)

    root = r"C:\Users\LENOVO\Downloads\Yolo\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"

    train_ds = DesertDataset(os.path.join(root, "train"), processor)
    val_ds   = DesertDataset(os.path.join(root, "val"), processor)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])

    best_iou = 0

    try:
        for epoch in range(CFG["epochs"]):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                pixel = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                out = model(pixel_values=pixel, labels=labels)
                loss = out.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Train Loss: {total_loss/len(train_loader):.4f}")

            # ===== VALIDATION =====
            model.eval()
            ious = []

            with torch.no_grad():
                for batch in val_loader:
                    pixel = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    out = model(pixel_values=pixel)
                    preds = F.interpolate(
                        out.logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    preds = preds.argmax(1)

                    ious.append(compute_miou(preds, labels))

            val_iou = np.mean(ious)
            print(f"Val mIoU: {val_iou:.4f}")

            # 🔥 SAVE EVERY EPOCH
            torch.save(model.state_dict(), f"segformer_epoch_{epoch+1}.pth")

            # ⭐ SAVE BEST MODEL
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), "best_model.pth")
                print("✅ Best model saved!")

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "interrupted_model.pth")
        print("⚠️ Model saved after interruption")


if __name__ == "__main__":
    train()