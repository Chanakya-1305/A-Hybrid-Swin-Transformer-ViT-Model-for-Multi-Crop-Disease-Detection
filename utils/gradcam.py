import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================= UNIFIED GRAD-CAM =================
class UnifiedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        # Use full_backward_hook for PyTorch 2.x compatibility
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Handle both 4D (B,C,H,W) from CNNs and 3D (B,tokens,C) from Transformers
        if gradients.dim() == 4:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1)
        elif gradients.dim() == 3:
            weights = gradients.mean(dim=1, keepdim=True)   # (B,1,C)
            cam = (weights * activations).sum(dim=2)        # (B,tokens)
            # reshape tokens back to 2D feature map (assume square)
            import math
            B, T, _ = activations.shape
            H = W = int(math.sqrt(T))
            if H * W == T:
                cam = cam.view(B, H, W)
            else:
                # fallback: keep flat and unsqueeze
                cam = cam.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected gradient dimensions: {gradients.dim()}")

        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()[0]
        if cam.ndim == 1:
            cam = cam.reshape(int(cam.shape[0]**0.5), -1)
        cam = cv2.resize(cam.astype(np.float32), (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

# ================= HEATMAP + OVERLAY =================
def generate_gradcam_overlay(
    image_path,
    cam,
    alpha=0.4,
    colormap=cv2.COLORMAP_JET
):
    """
    Produces X-ray style Grad-CAM visualization
    """

    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Convert CAM to heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return overlay, heatmap

# ================= END-TO-END USAGE =================
def run_gradcam(
    model,
    image_path,
    target_layer,
    save_path="gradcam_result.png",
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Load image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Init GradCAM
    gradcam = UnifiedGradCAM(model, target_layer)

    # Generate CAM
    cam = gradcam.generate(input_tensor)

    # Create colored overlay
    overlay, heatmap = generate_gradcam_overlay(image_path, cam)

    # Save result
    cv2.imwrite(save_path, overlay)

    print(f"Grad-CAM saved at: {save_path}")
    return save_path


def generate_gradcam(model, image_path, save_path):
    """Fault-tolerant GradCAM wrapper. Falls back to copying input image on any error."""
    import shutil
    try:
        target_layer = None

        # Prefer the patch embedding Conv2d (first Conv2d in the model) for transformers
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

        # Fallback: last Conv2d if first not found
        if target_layer is None:
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break

        if target_layer is None:
            raise ValueError("No Conv2d layer found in model.")

        run_gradcam(model, image_path, target_layer, save_path=save_path)

    except Exception as e:
        # If GradCAM fails for any reason, copy original image as fallback
        print(f"[GradCAM Warning] {e} - using original image as fallback.")
        shutil.copy(image_path, save_path)
