#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------------------------------------------------------------
# 1) SEED E CONFIGURAÇÃO BÁSICA
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# 2) HYPERPARÂMETROS AJUSTADOS PARA AdamW
# -----------------------------------------------------------------------------
num_epochs = 100
base_lr    = 0.0001      # menor para AdamW, evita steps muito grandes
weight_decay = 0.01      # um pouco maior para regularização L2
batch_size   = 12         # reduzindo batch_size se houver limitação de memória
EARLY_STOPPING_PATIENCE = 8
LR_SCHEDULER_PATIENCE   = 3

# -----------------------------------------------------------------------------
# 3) LOGGING
# -----------------------------------------------------------------------------
writer = SummaryWriter(log_dir='runs/fasterrcnn_experiment_adamw')

# -----------------------------------------------------------------------------
# 4) DATA PATHS E TRANSFORMS (AUGMENTATIONS SEGUROS)
# -----------------------------------------------------------------------------
IMG_DIR   = "/home/messyas/data/images"
ANN_TRAIN = "/home/messyas/data/annotations/4_class_splits/annotations_train.json"
ANN_VAL   = "/home/messyas/data/annotations/4_class_splits/annotations_val.json"

train_transforms = T.Compose([
    T.ToTensor(),

    # flips — não alteram geometria além de refletir horizontal/vertical
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),

    # ajustes de cor — brilho, contraste, saturação e matiz
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    # escala de cinza parcial — não muda bboxes
    T.RandomGrayscale(p=0.1),

    # borrão suave — não mexe nas coordenadas
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
])

val_transforms = T.Compose([
    T.ToTensor(),
])

# -----------------------------------------------------------------------------
# 5) DATASET E DATALOADER
# -----------------------------------------------------------------------------
class CocoLikeDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        from torchvision.datasets import CocoDetection
        self.coco = CocoDetection(img_dir, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anns = self.coco[idx]
        boxes, labels = [], []
        for obj in anns:
            x, y, w, h = obj['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(obj['category_id'])

        if boxes:
            boxes  = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)
        return img, {'boxes': boxes, 'labels': labels}

    def __len__(self):
        return len(self.coco)

train_ds = CocoLikeDataset(IMG_DIR, ANN_TRAIN, transforms=train_transforms)
val_ds   = CocoLikeDataset(IMG_DIR, ANN_VAL,   transforms=val_transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
    generator=torch.Generator().manual_seed(SEED)
)
valid_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# -----------------------------------------------------------------------------
# 6) MODELO E CABEÇA PARA 11 CLASSES
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model   = fasterrcnn_resnet50_fpn(weights=weights)

in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=11)
model.to(device)

# -----------------------------------------------------------------------------
# 7) OTIMIZADOR AdamW E SCHEDULER
# -----------------------------------------------------------------------------
params    = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=base_lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=LR_SCHEDULER_PATIENCE
)

# -----------------------------------------------------------------------------
# 8) LOOP DE TREINO + VALIDAÇÃO
# -----------------------------------------------------------------------------
best_val_loss = float('inf')
no_improve    = 0

for epoch in range(1, num_epochs + 1):
    # --- Treino ---
    model.train()
    running_loss = 0.0
    for imgs, tgts in train_loader:
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        optimizer.zero_grad()
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)

    # --- Validação (modo treino só para calcular loss) ---
    val_loss = 0.0
    model.train()              # <— necessário para retornar loss_dict
    with torch.no_grad():
        for imgs, tgts in valid_loader:
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            val_loss += sum(loss_dict.values()).item()
    model.eval()               # <— volta pro modo de avaliação padrão

    val_loss /= len(valid_loader)
    writer.add_scalar('Loss/val', val_loss, epoch)

    # --- Scheduler & Logging de LR ---
    scheduler.step(val_loss)
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('LearningRate', current_lr, epoch)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

    # --- Early Stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), "best_model_adamw.pth")
    else:
        no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping na epoca {epoch} (sem melhora em {EARLY_STOPPING_PATIENCE} epochs).")
            break

writer.close()
print("Treinamento concluído com AdamW.")


# In[ ]:




