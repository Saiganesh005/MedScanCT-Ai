import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import load_model, class_names

st.set_page_config(page_title="FastViT Lung CT Classifier", layout="centered")

@st.cache_resource
def get_model():
    return load_model()


model, device = get_model()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

st.title("🫁 FastViT — Lung CT Disease Classifier")
st.write("Upload a chest CT image (PNG/JPG) to get prediction.")

uploaded_file = st.file_uploader("Choose a CT Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", use_column_width=True)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    st.subheader("Prediction Result")
    st.success(f"🩺 **Diagnosis:** {class_names[pred]}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probs[0][i].item() * 100:.2f}%")
