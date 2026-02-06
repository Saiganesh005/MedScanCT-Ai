import requests
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import load_model, class_names

st.set_page_config(page_title="FastViT Lung CT Classifier", layout="centered")


@st.cache_resource
def get_model():
    return load_model(include_target_layer=True)


model, device, target_layer = get_model()

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

st.sidebar.title("🔐 Doctor Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    response = requests.post(
        "http://127.0.0.1:8000/token",
        data={"username": username, "password": password},
        timeout=10,
    )
    if response.status_code == 200:
        st.session_state["token"] = response.json()["access_token"]
        st.session_state["role"] = response.json()["role"]
        st.sidebar.success(f"Logged in as {st.session_state['role']}")
    else:
        st.sidebar.error("Invalid login")

token = st.session_state.get("token")

uploaded_file = st.file_uploader("Choose a CT Image", type=["png", "jpg", "jpeg"])


def generate_gradcam(model, target_layer, img_tensor):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_np = img_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization


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

    st.subheader("Explainable AI: Grad-CAM Heatmap")
    heatmap = generate_gradcam(model, target_layer, img_tensor)
    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
    st.write(
        "🔍 Highlighted regions show which parts of the lung influenced the model’s decision."
    )

st.divider()
st.subheader("Batch Prediction")

patient_name = st.text_input("Enter Patient Name")
uploaded_files = st.file_uploader(
    "Upload multiple CT images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if st.button("Run Batch Prediction") and uploaded_files and patient_name:
    if not token:
        st.warning("Please log in from the sidebar to run batch predictions.")
        st.stop()

    headers = {"Authorization": f"Bearer {token}"}
    files_payload = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
    response = requests.post(
        "http://127.0.0.1:8000/batch_predict/",
        params={"patient_name": patient_name},
        files=files_payload,
        headers=headers,
        timeout=30,
    )
    if response.status_code != 200:
        st.error("Batch prediction failed. Check credentials or server logs.")
        st.stop()

    results = response.json().get("results", [])

    st.subheader("Batch Results")
    for result in results:
        st.write(
            f"📁 {result['filename']} → {result['prediction']} ({result['confidence']}%)"
        )

st.subheader("Stored Patient Records")
if st.button("Load Records"):
    if not token:
        st.warning("Please log in from the sidebar to view patient records.")
        st.stop()

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        "http://127.0.0.1:8000/get_records/",
        headers=headers,
        timeout=10,
    )
    if response.status_code != 200:
        st.error("Unable to load records. Check credentials or server logs.")
        st.stop()

    records = response.json().get("records", [])
    st.dataframe(records)
