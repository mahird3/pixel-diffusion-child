import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import requests
import replicate

from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification

# Add your custom model folder
sys.path.append("src")
from LatentDiffusion import LatentDiffusionConditional

# Face detection (insightface)
import insightface

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load diffusion model
ckpt_path = hf_hub_download(
    repo_id="mahird3/pixel-diffusion-child",
    filename="pixel-diffusion-final.ckpt",
    repo_type="model"
)
model = LatentDiffusionConditional.load_from_checkpoint(ckpt_path, train_dataset=None)
model.eval().cuda()

# Load age classification model (still used for flexibility, but not applying SAM)
age_model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier").eval().cuda()
age_processor = ViTImageProcessor.from_pretrained("nateraw/vit-age-classifier")

# Set up face detector
detector = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=0)

def extract_main_face(pil_img: Image.Image, margin_ratio=0.2) -> Image.Image:
    img = np.array(pil_img)
    faces = detector.get(img)
    if not faces:
        print("⚠️ No face detected — using full image.")
        return pil_img

    face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
    x1, y1, x2, y2 = map(int, face.bbox)

    h_margin = int((y2 - y1) * margin_ratio)
    w_margin = int((x2 - x1) * margin_ratio)
    y1 = max(0, y1 - h_margin)
    y2 = min(img.shape[0], y2 + h_margin)
    x1 = max(0, x1 - w_margin)
    x2 = min(img.shape[1], x2 + w_margin)

    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cropped)

def upload_image_to_imgbb(pil_image: Image.Image) -> str:
    api_key = os.getenv("IMGBB_API_KEY")
    if not api_key:
        raise ValueError("IMGBB_API_KEY not set.")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    buffered.seek(0)
    files = {"image": buffered}
    params = {"key": api_key}
    response = requests.post("https://api.imgbb.com/1/upload", files=files, params=params)
    return response.json()["data"]["url"]

def predict(father_image: Image.Image, mother_image: Image.Image, gender: str) -> Image.Image:
    # Face detection and resize
    father_pil = extract_main_face(father_image).resize((128, 128))
    mother_pil = extract_main_face(mother_image).resize((128, 128))

    f_tensor = transform(father_pil).unsqueeze(0).cuda()
    m_tensor = transform(mother_pil).unsqueeze(0).cuda()
    gender_tensor = torch.tensor([0 if gender.lower() == "boy" else 1]).cuda()

    with torch.no_grad():
        child_tensor = model.generate_from_parents(f_tensor[0], m_tensor[0], gender_tensor.item())
        codeformer_input = F.interpolate(child_tensor.unsqueeze(0) * 2 - 1, size=(512, 512), mode="bilinear")
        result_tensor = codeformer_input[0].clamp(-1, 1)
        result_np = ((result_tensor + 1) / 2).cpu().permute(1, 2, 0).numpy()
        result_pil = Image.fromarray((result_np * 255).astype("uint8"))

    # Enhance with CodeFormer via Replicate
    image_url = upload_image_to_imgbb(result_pil)
    enhanced_url = replicate.run(
        "sczhou/codeformer:8b6d72a9cfedf8e3d6ad9596dfb219209b3704023d2202b66a5c1e7316b5efde",
        input={
            "image": image_url,
            "face_upsample": True,
            "codeformer_fidelity": 0.7
        }
    )
    response = requests.get(enhanced_url)
    return Image.open(BytesIO(response.content))
