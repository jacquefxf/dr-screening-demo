"""
Diabetic Retinopathy Screening Demo — Team 6thSense
National AI Competition 2026

Disclaimer: This is a research prototype. NOT a medical diagnosis tool.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# ============================================================
# Config
# ============================================================
MODEL_NAME = "convnext_tiny"
IMAGE_SIZE = 512
NUM_CLASSES = 5
CLASS_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "PDR"}
CLASS_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c", 4: "#9b59b6"}
CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Small areas of balloon-like swelling in the retina's blood vessels (microaneurysms).",
    2: "Some blood vessels are blocked, with small bleeds and protein/fat deposits visible.",
    3: "Many blood vessels are blocked, signalling the retina to grow new (fragile) vessels.",
    4: "New blood vessels have grown, which are fragile and may leak or bleed. Most advanced stage.",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WEIGHT_PATH = "best_model_fold0_fp16.pth"
GDRIVE_FILE_ID = "1CC6CwzlB_4xJxYZrM3z-XJgeuOCldckU"


# ============================================================
# Download model from Google Drive if not present
# ============================================================
def download_model_weights():
    if os.path.exists(WEIGHT_PATH):
        return True
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"
    try:
        import urllib.request
        with st.spinner("Downloading model weights (first time only, ~55MB)..."):
            urllib.request.urlretrieve(url, WEIGHT_PATH)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False


# ============================================================
# Load model (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    if not download_model_weights():
        return None
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu")
    # Convert fp16 weights back to fp32 for inference
    state_dict = {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================
# Prediction
# ============================================================
def predict(model, image_np):
    """Run prediction with TTA (4 flips)."""
    transform = get_transform()
    transformed = transform(image=image_np)["image"].unsqueeze(0)

    with torch.no_grad():
        p1 = F.softmax(model(transformed).float(), dim=1)
        p2 = F.softmax(model(torch.flip(transformed, [3])).float(), dim=1)
        p3 = F.softmax(model(torch.flip(transformed, [2])).float(), dim=1)
        p4 = F.softmax(model(torch.flip(transformed, [2, 3])).float(), dim=1)

    probs = ((p1 + p2 + p3 + p4) / 4).squeeze().numpy()
    pred_class = int(probs.argmax())
    confidence = float(probs.max())
    return pred_class, confidence, probs


# ============================================================
# Grad-CAM
# ============================================================
def generate_gradcam(model, image_np, target_class):
    transform = get_transform()
    input_tensor = transform(image=image_np)["image"].unsqueeze(0)

    target_layers = [model.stages[-1].blocks[-1].conv_dw]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    img_resized = np.array(Image.fromarray(image_np).resize((IMAGE_SIZE, IMAGE_SIZE)))
    img_float = img_resized.astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    del cam
    return cam_image


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="DR Screening — Team 6thSense",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ Diabetic Retinopathy Screening")
st.markdown("**Team 6thSense** — Elijah, KaiXi, Garet, Isaac · National AI Competition 2026")

st.warning(
    "⚠️ **Disclaimer:** This is a research prototype built for the National AI Competition 2026. "
    "It is **NOT** a medical diagnosis tool and should not be used for clinical decision-making. "
    "Always consult a qualified ophthalmologist for DR screening."
)

# Load model
model = load_model()

if model is None:
    st.stop()

# File upload
uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["png", "jpg", "jpeg"],
    help="Upload a retinal fundus photograph for DR severity grading."
)

if uploaded_file is not None:
    # Load image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image_pil)

    # Predict
    with st.spinner("Analyzing image..."):
        pred_class, confidence, probs = predict(model, image_rgb)
        cam_image = generate_gradcam(model, image_rgb, pred_class)

    # Layout: two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_rgb, use_container_width=True)

        st.subheader("Grad-CAM Heatmap")
        st.image(cam_image, use_container_width=True)
        st.caption("Highlights regions the model focused on for its prediction.")

    with col2:
        st.subheader("Prediction")

        # Result card
        color = CLASS_COLORS[pred_class]
        st.markdown(
            f'<div style="background-color:{color}20; border-left:5px solid {color}; '
            f'padding:15px; border-radius:5px; margin-bottom:15px;">'
            f'<h2 style="color:{color}; margin:0;">Class {pred_class}: {CLASS_NAMES[pred_class]}</h2>'
            f'<p style="margin:5px 0 0 0; font-size:16px;">Confidence: {confidence:.1%}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"**Description:** {CLASS_DESCRIPTIONS[pred_class]}")

        # Probability bars
        st.subheader("Class Probabilities")
        for i in range(NUM_CLASSES):
            prob = float(probs[i])
            label = f"Class {i}: {CLASS_NAMES[i]}"
            st.progress(prob, text=f"{label} — {prob:.1%}")

        # Model info
        with st.expander("Model Details"):
            st.markdown(f"""
            - **Architecture:** ConvNeXt-Tiny (28.6M params)
            - **Input Resolution:** {IMAGE_SIZE}×{IMAGE_SIZE}
            - **Training:** 5-Fold CV, best fold used for demo
            - **Inference:** TTA (4 geometric flips averaged)
            - **OOF Macro F1:** 0.7636
            - **OOF Accuracy:** 85.69%
            """)

else:
    # Show info when no image uploaded
    st.info("👆 Upload a retinal fundus image to get started.")

    st.markdown("### About This Tool")
    st.markdown(
        "This demo uses a ConvNeXt-Tiny deep learning model trained on 4,939 retinal fundus images "
        "to classify diabetic retinopathy into 5 severity grades. The model was developed as part of "
        "the National AI Competition 2026 Computing Track."
    )

    st.markdown("### DR Severity Grades")
    for i in range(NUM_CLASSES):
        color = CLASS_COLORS[i]
        st.markdown(
            f"**<span style='color:{color}'>Class {i} — {CLASS_NAMES[i]}:</span>** "
            f"{CLASS_DESCRIPTIONS[i]}",
            unsafe_allow_html=True
        )
