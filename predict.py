import base64
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import replicate
from torchvision import transforms

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def decode_base64_image(data):
    return Image.open(BytesIO(base64.b64decode(data)))

def encode_base64_image(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def upload_to_imgbb(pil_img):
    """Temporarily host image to get a public URL for Replicate API"""
    imgbb_key = os.getenv("IMGBB_API_KEY")
    if not imgbb_key:
        raise ValueError("IMGBB_API_KEY environment variable is not set")
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    buffered.seek(0)
    files = {"image": buffered}
    response = requests.post(
        "https://api.imgbb.com/1/upload",
        params={"key": imgbb_key},
        files=files
    )
    return response.json()["data"]["url"]

def predict(father_base64: str, mother_base64: str, gender: str) -> str:
    # Decode images
    father_img = decode_base64_image(father_base64)
    mother_img = decode_base64_image(mother_base64)

    # Preprocess for torch
    f_tensor = transform(father_img).unsqueeze(0).cuda()
    m_tensor = transform(mother_img).unsqueeze(0).cuda()
    gender_tensor = torch.tensor([0 if gender.lower() == 'boy' else 1]).cuda()

    # Dummy generation logic — blend parent features
    mixed = (f_tensor + m_tensor) / 2

    # Normalize to [-1, 1]
    img_tensor = F.interpolate(mixed * 2 - 1, size=(512, 512), mode='bilinear', align_corners=False)
    img = img_tensor[0].cpu().clamp(-1, 1)
    img = ((img + 1) / 2).permute(1, 2, 0).numpy()
    img_pil = Image.fromarray((img * 255).astype("uint8"))

    # Upload to imgbb
    img_url = upload_to_imgbb(img_pil)

    # Call CodeFormer on Replicate
    result_url = replicate.run(
        "sczhou/codeformer:8b6d72a9cfedf8e3d6ad9596dfb219209b3704023d2202b66a5c1e7316b5efde",
        input={
            "image": img_url,
            "face_upsample": True,
            "codeformer_fidelity": 0.7
        }
    )

    # Fetch result and return as base64
    final_img = Image.open(requests.get(result_url, stream=True).raw)
    return encode_base64_image(final_img)