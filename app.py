"""
Diabetic Retinopathy Screening Demo - Team 6thSense
National AI Competition 2026
"""

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import os

MODEL_NAME = "convnext_tiny"
IMAGE_SIZE = 512
NUM_CLASSES = 5
CLASS_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "PDR"}
CLASS_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c", 4: "#9b59b6"}
CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Small areas of balloon-like swelling in the retina\'s blood vessels (microaneurysms).",
    2: "Some blood vessels are blocked, with small bleeds and protein/fat deposits visible.",
    3: "Many blood vessels are blocked, signalling the retina to grow new (fragile) vessels.",
    4: "New blood vessels have grown, which are fragile and may leak or bleed. Most advanced stage.",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
WEIGHT_PATH = "best_model_fold0_fp16.pth"
GDRIVE_FILE_ID = "1IjJAOJsOCImmgnsfU5od06R9YdUL0XD1"

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def download_model_weights():
    if os.path.exists(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) > 1000000:
        return True
    if os.path.exists(WEIGHT_PATH):
        os.remove(WEIGHT_PATH)
    try:
        import gdown
        with st.spinner("Downloading model weights (first time only, ~55MB)..."):
            gdown.download(id=GDRIVE_FILE_ID, output=WEIGHT_PATH, quiet=False)
        if os.path.exists(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) > 1000000:
            return True
        else:
            st.error("Download failed. Check Google Drive sharing settings.")
            return False
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False


@st.cache_resource
def load_model():
    if not download_model_weights():
        return None
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    state_dict = {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, image_pil):
    input_tensor = inference_transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        p1 = F.softmax(model(input_tensor).float(), dim=1)
        p2 = F.softmax(model(torch.flip(input_tensor, [3])).float(), dim=1)
        p3 = F.softmax(model(torch.flip(input_tensor, [2])).float(), dim=1)
        p4 = F.softmax(model(torch.flip(input_tensor, [2, 3])).float(), dim=1)
    probs = ((p1 + p2 + p3 + p4) / 4).squeeze().numpy()
    pred_class = int(probs.argmax())
    confidence = float(probs.max())
    return pred_class, confidence, probs


def generate_activation_map(model, image_pil):
    input_tensor = inference_transform(image_pil).unsqueeze(0)
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    handle = model.stages[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()

    if len(activations) == 0:
        return None

    act = activations[0].squeeze(0)
    heatmap = act.mean(dim=0).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0

    img_resized = np.array(image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

    colored_heatmap = np.zeros((*heatmap_resized.shape, 3), dtype=np.float32)
    colored_heatmap[:, :, 0] = heatmap_resized
    colored_heatmap[:, :, 1] = heatmap_resized * 0.3

    overlay = img_resized * 0.6 + colored_heatmap * 0.4
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


st.set_page_config(page_title="DR Screening - Team 6thSense", page_icon="eye", layout="wide")
st.title("Diabetic Retinopathy Screening")
st.markdown("**Team 6thSense** - Elijah, KaiXi, Garet, Isaac | National AI Competition 2026")
st.warning(
    "Disclaimer: This is a research prototype built for the National AI Competition 2026. "
    "It is NOT a medical diagnosis tool and should not be used for clinical decision-making. "
    "Always consult a qualified ophthalmologist for DR screening."
)

model = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing image..."):
        pred_class, confidence, probs = predict(model, image_pil)
        heatmap = generate_activation_map(model, image_pil)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_pil, use_container_width=True)
        if heatmap is not None:
            st.subheader("Activation Heatmap")
            st.image(heatmap, use_container_width=True)
            st.caption("Highlights regions the model focused on for its prediction.")

    with col2:
        st.subheader("Prediction")
        color = CLASS_COLORS[pred_class]
        st.markdown(
            f"<div style=\"background-color:{color}20; border-left:5px solid {color}; "
            f"padding:15px; border-radius:5px; margin-bottom:15px;\">"
            f"<h2 style=\"color:{color}; margin:0;\">Class {pred_class}: {CLASS_NAMES[pred_class]}</h2>"
            f"<p style=\"margin:5px 0 0 0; font-size:16px;\">Confidence: {confidence:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Description:** {CLASS_DESCRIPTIONS[pred_class]}")
        st.subheader("Class Probabilities")
        for i in range(NUM_CLASSES):
            prob = float(probs[i])
            label = f"Class {i}: {CLASS_NAMES[i]}"
            st.progress(prob, text=f"{label} - {prob:.1%}")
        with st.expander("Model Details"):
            st.markdown(
                f"- **Architecture:** ConvNeXt-Tiny (28.6M params)\n"
                f"- **Input Resolution:** {IMAGE_SIZE}x{IMAGE_SIZE}\n"
                f"- **Training:** 5-Fold CV, best fold used for demo\n"
                f"- **Inference:** TTA (4 geometric flips averaged)\n"
                f"- **OOF Macro F1:** 0.7636\n"
                f"- **OOF Accuracy:** 85.69%"
            )
else:
    st.info("Upload a retinal fundus image to get started.")
    st.markdown("### About This Tool")
    st.markdown(
        "This demo uses a ConvNeXt-Tiny deep learning model trained on 4,939 retinal fundus images "
        "to classify diabetic retinopathy into 5 severity grades."
    )
    st.markdown("### DR Severity Grades")
    for i in range(NUM_CLASSES):
        color = CLASS_COLORS[i]
        st.markdown(
            f"**Class {i} - {CLASS_NAMES[i]}:** {CLASS_DESCRIPTIONS[i]}"
        )
