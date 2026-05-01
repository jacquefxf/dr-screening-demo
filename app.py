"""
DR Screening Demo - Team 6thSense
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
    1: "Small areas of balloon-like swelling in the retina blood vessels (microaneurysms).",
    2: "Some blood vessels are blocked, with small bleeds and protein/fat deposits visible.",
    3: "Many blood vessels are blocked, signalling the retina to grow new (fragile) vessels.",
    4: "New blood vessels have grown, which are fragile and may leak or bleed. Most advanced stage.",
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
WEIGHT_PATH = "best_model_fold0_fp16.pth"
GDRIVE_FILE_ID = "1IjJAOJsOCImmgnsfU5od06R9YdUL0XD1"

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def download_model_weights():
    if os.path.exists(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) > 1000000:
        return True
    if os.path.exists(WEIGHT_PATH):
        os.remove(WEIGHT_PATH)
    try:
        import gdown
        with st.spinner("Downloading model weights (~55MB)..."):
            gdown.download(id=GDRIVE_FILE_ID, output=WEIGHT_PATH, quiet=False)
        if os.path.exists(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) > 1000000:
            return True
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
    inp = inference_transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        p1 = F.softmax(model(inp).float(), dim=1)
        p2 = F.softmax(model(torch.flip(inp, [3])).float(), dim=1)
        p3 = F.softmax(model(torch.flip(inp, [2])).float(), dim=1)
        p4 = F.softmax(model(torch.flip(inp, [2, 3])).float(), dim=1)
    probs = ((p1 + p2 + p3 + p4) / 4).squeeze().numpy()
    return int(probs.argmax()), float(probs.max()), probs


def generate_gradcam(model, image_pil, target_class):
    """Grad-CAM using manual gradient computation - no external deps needed."""
    inp = inference_transform(image_pil).unsqueeze(0)
    inp.requires_grad = False

    # hook the last conv block
    activations = []
    gradients = []

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer = model.stages[-1].blocks[-1].conv_dw
    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    # forward + backward for target class
    inp_grad = inp.clone().requires_grad_(True)
    output = model(inp_grad)
    model.zero_grad()
    target_score = output[0, target_class]
    target_score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    if not activations or not gradients:
        return None

    # grad-cam: weight activations by mean gradient per channel
    act = activations[0].squeeze(0)   # (C, H, W)
    grad = gradients[0].squeeze(0)    # (C, H, W)
    weights = grad.mean(dim=(1, 2))   # (C,)
    cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.numpy()

    # resize to image size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = np.array(cam_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0

    # create RGB heatmap (blue=low, red=high)
    heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    heatmap[:, :, 0] = cam_resized                    # red
    heatmap[:, :, 1] = cam_resized * 0.4              # green tint
    heatmap[:, :, 2] = (1 - cam_resized) * 0.3        # blue for low activation

    # overlay on original image
    img_resized = np.array(image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    overlay = img_resized * 0.5 + heatmap * 0.5
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

    with st.spinner("Analyzing..."):
        pred_class, confidence, probs = predict(model, image_pil)
        gradcam_img = generate_gradcam(model, image_pil, pred_class)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_pil, use_container_width=True)
        if gradcam_img is not None:
            st.subheader("Grad-CAM Heatmap")
            st.image(gradcam_img, use_container_width=True)
            st.caption("Red regions = where the model focused. Highlights lesions and abnormalities.")

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
            st.progress(prob, text=f"Class {i}: {CLASS_NAMES[i]} - {prob:.1%}")
        with st.expander("Model Details"):
            st.markdown(
                f"- **Architecture:** ConvNeXt-Tiny (28.6M params)\n"
                f"- **Input Resolution:** {IMAGE_SIZE}x{IMAGE_SIZE}\n"
                f"- **Training:** 5-Fold CV, best fold used for demo\n"
                f"- **Inference:** TTA (4 geometric flips averaged)\n"
                f"- **OOF Macro F1:** 0.764\n"
                f"- **Mean Fold F1:** 0.79"
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
        st.markdown(f"**Class {i} - {CLASS_NAMES[i]}:** {CLASS_DESCRIPTIONS[i]}")
