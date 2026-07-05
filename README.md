# Crop Disease Detection Web App

A Flask-based web application for crop disease classification using a hybrid Swin Transformer + ViT model, with Grad-CAM visualization support.

## Features

- User authentication (signup/login/logout) with SQLite
- Upload crop leaf images for prediction
- Disease class prediction with confidence score
- Confidence-based response:
  - Confidence > 65%: disease result + Grad-CAM visualization
  - Confidence <= 65%: fallback message
- Simple web interface with dashboard and result pages

## Project Structure

```
cropdiseasedetection/
|-- app.py
|-- models/
|   `-- best_hybrid_swin_vit_crop_disease.pth
|-- static/
|   |-- uploads/
|   `-- results/
|-- templates/
|   |-- home.html
|   |-- login.html
|   |-- signup.html
|   |-- dashboard.html
|   |-- resultpage.html
|   `-- aboutus.html
`-- utils/
    |-- predict.py
    `-- gradcam.py
```

## Requirements

- Python 3.10+
- pip
- A valid model file at:
  - `models/best_hybrid_swin_vit_crop_disease.pth`

> Note: For GPU acceleration, install a CUDA-compatible PyTorch build if your system supports it.

## Installation

### 1. Clone or open the project

```bash
git clone <your-repo-url>
cd cropdiseasedetection
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install Flask Werkzeug torch torchvision timm pillow numpy opencv-python
```

## Run the Application

From the project root:

```bash
python app.py
```

The app starts on:

- `http://127.0.0.1:5001`

(Port is configurable via the `PORT` environment variable.)

## How to Use

1. Open the app in your browser.
2. Create an account using **Sign Up**.
3. Log in.
4. Go to the dashboard and upload an image.
5. View the prediction result and confidence.
6. If confidence is high, a Grad-CAM image is generated in `static/results/`.

## Output Files

- Uploaded images: `static/uploads/`
- Grad-CAM result images: `static/results/`
- SQLite database file: `crop_disease_detection.db`

## Troubleshooting

- Model load error:
  - Ensure `models/best_hybrid_swin_vit_crop_disease.pth` exists and matches the architecture in `utils/predict.py`.
- Import errors:
  - Re-check your virtual environment activation and dependency installation.
- Port already in use:
  - Set another port before running.

Windows (PowerShell):

```powershell
$env:PORT=5002
python app.py
```

macOS/Linux:

```bash
export PORT=5002
python app.py
```

## Security Note

This project is for learning/demo use. Before production use, update secret management (for example, move Flask secret key to environment variables) and harden authentication/session settings.
