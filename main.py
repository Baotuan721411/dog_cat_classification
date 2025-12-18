import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

MODEL_PATH = "efficientnet_b1_dogs_cats.pth"
CLASS_NAMES = ["Cat", "Dog"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = timm.create_model(
        "efficientnet_b1",
        pretrained=False,
        num_classes=2
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =============================
# IMAGE TRANSFORM (GIỐNG TRAIN)
# =============================
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Dog vs Cat", layout="centered")
st.title("🐶🐱 Dog vs Cat Classifier (EfficientNet-B1)")
st.write("Upload ảnh để mô hình dự đoán")

uploaded_file = st.file_uploader(
    "📂 Chọn ảnh",
    type=["jpg", "jpeg", "png"]
)

# =============================
# PREDICT
# =============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã upload", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    st.markdown("### 📌 Kết quả dự đoán")
    st.success(f"👉 **{CLASS_NAMES[pred_idx]}**")

    st.markdown("### 📊 Xác suất")
    st.write({
        "Cat": float(probs[0][0]),
        "Dog": float(probs[0][1])
    })
