# ==============================
# 1. Mount Google Drive
# ==============================
from google.colab import drive
drive.mount('/content/drive')

# ==============================
# 2. Install dependencies
# ==============================
!pip install timm tqdm scikit-learn

# ==============================
# 3. Imports
# ==============================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report

# ==============================
# 4. Paths
# ==============================
DATA_DIR  = "/content/drive/MyDrive/datasets"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR   = os.path.join(DATA_DIR, "valid")

MODEL_DIR = "/content/drive/MyDrive/CropDiseaseModels"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# 5. Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 6. Transforms
# ==============================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 7. Dataset
# ==============================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

# Ensure class mapping same
val_dataset.class_to_idx = train_dataset.class_to_idx

num_classes = len(train_dataset.classes)

print("Classes:", num_classes)
print(train_dataset.classes)

# ==============================
# 8. DataLoader (optimized)
# ==============================
train_loader = DataLoader(
    train_dataset,
    batch_size=8,      # safer for Colab
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ==============================
# 9. Hybrid Model
# ==============================
class HybridSwinViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )

        self.fc = nn.Linear(768 + 768, num_classes)

    def forward(self, x):
        vit_feat  = self.vit(x)
        swin_feat = self.swin(x)
        combined  = torch.cat((vit_feat, swin_feat), dim=1)
        return self.fc(combined)

model = HybridSwinViT(num_classes).to(device)

# Freeze backbones (important)
for p in model.vit.parameters():
    p.requires_grad = False

for p in model.swin.parameters():
    p.requires_grad = False

# ==============================
# 10. Loss + Optimizer + AMP (fixed)
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scaler = torch.amp.GradScaler("cuda")

# ==============================
# 11. Training Functions
# ==============================
def train_epoch(model, loader):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def validate_epoch(model, loader):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total

# ==============================
# 12. Training Loop
# ==============================
EPOCHS = 5
best_val_acc = 0.0
best_path = f"{MODEL_DIR}/best_hybrid_swin_vit_crop_disease.pth"

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc     = validate_epoch(model, val_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print("✅ Best model saved:", best_path)

# ==============================
# 13. Evaluation
# ==============================
print("\nLoading best model...")
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

# ==============================
# 14. Download Model
# ==============================
from google.colab import files
files.download(best_path)