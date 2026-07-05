import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import os

# ==============================
# Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Class Names (IMPORTANT)
# Same order used during training
# ==============================
CLASS_NAMES = [
'Maize_Cercosporaleafspot',
'Maize_Commonrust',
'Maize_Earrot',
'Maize_GrayLeafSpot',
'Maize__NorthernLeafBlight',
'Maize_healthy',
'Rice_Bacterialleafblight',
'Rice_Blast',
'Rice_Brownspot',
'Rice_Hispa',
'Rice_Leafsmut',
'Rice_NeckBlast',
'Rice_SheathBlight',
'Rice_Tungro',
'Rice_healthy',
'Rice_leafblast',
'Rice_leafscald',
'Rice_narrowbrownspot',
'Tomato_Bacterialspot',
'Tomato_Brownspot',
'Tomato_Earlyblight',
'Tomato_Lateblight',
'Tomato_LeafCurl',
'Tomato_LeafMold',
'Tomato_Septorialeafspot',
'Tomato_SpidermitesTwospottedspidermite',
'Tomato_TargetSpot',
'Tomato_TomatoYellowLeafCurlVirus',
'Tomato_Tomatomosaicvirus',
'Tomato_healthy'
]

num_classes = len(CLASS_NAMES)

# ==============================
# Model Architecture
# ==============================
class HybridSwinViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0
        )

        self.fc = nn.Linear(768 + 768, num_classes)

    def forward(self, x):
        vit_feat = self.vit(x)
        swin_feat = self.swin(x)

        combined = torch.cat((vit_feat, swin_feat), dim=1)

        return self.fc(combined)


# ==============================
# Load Model
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_hybrid_swin_vit_crop_disease.pth")

model = HybridSwinViT(num_classes).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


# ==============================
# Image Transform
# ==============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==============================
# Prediction Function
# ==============================
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    class_name = CLASS_NAMES[predicted.item()]

    conf = confidence.item() * 100

    return class_name, round(conf, 2)